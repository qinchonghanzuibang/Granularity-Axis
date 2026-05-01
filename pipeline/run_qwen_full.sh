#!/bin/bash
#
# One-click full pipeline for Qwen3-8B (or compatible HF/vLLM model).
# This script runs the end-to-end experiment flow:
#   - Main pipeline: roles -> responses -> activations -> vectors -> axis
#   - Supplementary: representation, subgroups, steering sweeps, evaluation, score filtering
#   - Steering robustness: truncation + micro-targeted reruns + summary (optional judge scoring)
#
# Usage:
#   cd /path/to/granularity_official
#   bash pipeline/run_qwen_full.sh
#
# Common overrides:
#   MODEL=/path/to/Qwen3-8B bash pipeline/run_qwen_full.sh
#   OUTPUT_DIR=outputs/qwen3-8b bash pipeline/run_qwen_full.sh
#   TENSOR_PARALLEL_SIZE=2 bash pipeline/run_qwen_full.sh
#   JUDGE_MODELS="gpt-4.1-mini" OPENAI_API_KEY=... bash pipeline/run_qwen_full.sh
#

set -e

source activate granularity

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

cd "$PROJECT_DIR"

# ================= Configuration =================
MODEL="${MODEL:-/mnt/dhwfile/raise/user/qinchonghan/models/Qwen3-8B}"
MODEL_SLUG="${MODEL_SLUG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/ ' '__')}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_SLUG}}"

ROLES_DIR="${ROLES_DIR:-data/roles/instructions}"
ENTITIES_FILE="${ENTITIES_FILE:-data/entities.json}"
QUESTIONS_FILE="${QUESTIONS_FILE:-data/extraction_questions.jsonl}"
METADATA_FILE="${METADATA_FILE:-data/role_metadata.json}"

# Main pipeline params
QUESTION_COUNT="${QUESTION_COUNT:-240}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TOKENS="${MAX_TOKENS:-1536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
ACTIVATION_POOLING="${ACTIVATION_POOLING:-mean}"
POOLING_K_TOKENS="${POOLING_K_TOKENS:-16}"
THINKING="${THINKING:-false}"

# Optional role-play judge (used by score filtering batch)
ROLE_JUDGE_MODEL="${ROLE_JUDGE_MODEL:-gpt-4.1-mini}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-25}"
JUDGE_RPS="${JUDGE_RPS:-50}"
JUDGE_MODELS="${JUDGE_MODELS:-gpt-4.1-mini}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-${API_KEY:-}}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-${BASE_HOST:-}}}"

# Score filtering
MIN_SCORE_STRICT="${MIN_SCORE_STRICT:-3}"
MIN_SCORE_LOOSE="${MIN_SCORE_LOOSE:-2}"

# Axis params
TARGET_LAYER="${TARGET_LAYER:-18}"
MICRO_LEVELS="${MICRO_LEVELS:-1 2}"
MACRO_LEVELS="${MACRO_LEVELS:-4 5}"
MIN_COUNT="${MIN_COUNT:-30}"

# Steering params (paper default: alpha in {-4, 0, +4}, greedy)
STEERING_COEFFS="${STEERING_COEFFS:--4.0 0.0 4.0}"
BASELINE_COEFFS="${BASELINE_COEFFS:--4.0 0.0 4.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_PROMPTS_GENERIC="${MAX_PROMPTS_GENERIC:-40}"
MAX_PROMPTS_BASELINE="${MAX_PROMPTS_BASELINE:-40}"
MAX_PROMPTS_SAMPLED="${MAX_PROMPTS_SAMPLED:-40}"

# Steering-robustness params
TRUNCATION_SHORT_TOKENS="${TRUNCATION_SHORT_TOKENS:-256}"
TRUNCATION_LONG_TOKENS="${TRUNCATION_LONG_TOKENS:-1536}"
PRIMARY_RISK_JUDGE_MODEL="${PRIMARY_RISK_JUDGE_MODEL:-gpt-5.4-mini}"

# Which parts to run
RUN_SCORE_FILTERING="${RUN_SCORE_FILTERING:-0}"
RUN_STEERING_ROBUSTNESS="${RUN_STEERING_ROBUSTNESS:-0}"
RUN_DECODING_SENSITIVITY="${RUN_DECODING_SENSITIVITY:-0}"
RUN_MICRO_TARGETED="${RUN_MICRO_TARGETED:-1}"
# =================================================

if [ -n "$JUDGE_API_KEY" ]; then
  export OPENAI_API_KEY="$JUDGE_API_KEY"
fi
if [ -n "$JUDGE_BASE_URL" ]; then
  OPENAI_BASE_URL="${JUDGE_BASE_URL%/}"
  if [[ "$OPENAI_BASE_URL" != */v1 ]]; then
    OPENAI_BASE_URL="${OPENAI_BASE_URL}/v1"
  fi
  export OPENAI_BASE_URL
fi

echo "============================================================"
echo "  Full Flow (Qwen): Granularity Axis"
echo "  Model:      $MODEL"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

echo ""
echo "=== Stage 0: Prepare role files ==="
python pipeline/0_prepare_roles.py \
  --entities_file "$ENTITIES_FILE" \
  --output_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE"

echo ""
echo "=== Stage 1: Generate role responses (vLLM) ==="
python pipeline/1_generate.py \
  --model "$MODEL" \
  --roles_dir "$ROLES_DIR" \
  --questions_file "$QUESTIONS_FILE" \
  --output_dir "$OUTPUT_DIR/responses" \
  --question_count "$QUESTION_COUNT" \
  --max_model_len "$MAX_MODEL_LEN" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  ${TENSOR_PARALLEL_SIZE:+--tensor_parallel_size "$TENSOR_PARALLEL_SIZE"}

echo ""
echo "=== Stage 2: Extract activations ==="
python pipeline/2_activations.py \
  --model "$MODEL" \
  --responses_dir "$OUTPUT_DIR/responses" \
  --output_dir "$OUTPUT_DIR/activations" \
  --batch_size "$BATCH_SIZE" \
  --max_length "$MAX_MODEL_LEN" \
  --pooling "$ACTIVATION_POOLING" \
  --k_tokens "$POOLING_K_TOKENS" \
  --thinking "$THINKING"

echo ""
echo "=== Stage 3: Compute vectors ==="
python pipeline/4_vectors.py \
  --activations_dir "$OUTPUT_DIR/activations" \
  --roles_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE" \
  --output_dir "$OUTPUT_DIR/vectors" \
  --min_count "$MIN_COUNT" \
  --no_scores

echo ""
echo "=== Stage 4: Compute axis ==="
python pipeline/5_axis.py \
  --vectors_dir "$OUTPUT_DIR/vectors" \
  --output_dir "$OUTPUT_DIR/axis" \
  --micro_levels $MICRO_LEVELS \
  --macro_levels $MACRO_LEVELS \
  --target_layer "$TARGET_LAYER"

echo ""
echo "=== Stage 5: Representation suite ==="
python pipeline/7_representation_suite.py \
  --vectors_dir "$OUTPUT_DIR/vectors" \
  --activations_dir "$OUTPUT_DIR/activations" \
  --roles_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE" \
  --output_dir "$OUTPUT_DIR/analysis/representation" \
  --target_layer "$TARGET_LAYER" \
  --micro_levels $MICRO_LEVELS \
  --macro_levels $MACRO_LEVELS \
  --role_holdout_fraction 0.33 \
  --seed 42

echo ""
echo "=== Stage 6: Subgroup analysis ==="
python pipeline/10_subgroup_analysis.py \
  --vectors_dir "$OUTPUT_DIR/vectors" \
  --activations_dir "$OUTPUT_DIR/activations" \
  --roles_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE" \
  --output_dir "$OUTPUT_DIR/analysis/subgroups" \
  --target_layer "$TARGET_LAYER" \
  --micro_levels $MICRO_LEVELS \
  --macro_levels $MACRO_LEVELS

echo ""
echo "=== Stage 7: Steering sweeps ==="
mkdir -p "$OUTPUT_DIR/steering"

# Generic prompts (paper main): alpha in {-4,0,+4} on 40 prompts.
python pipeline/6_steer.py \
  --model "$MODEL" \
  --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
  --vectors_dir "$OUTPUT_DIR/vectors" \
  --roles_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE" \
  --layers "$TARGET_LAYER" \
  --coeffs $STEERING_COEFFS \
  --directions granularity \
  --prompts_file data/steering_prompts.jsonl \
  --max_prompts "$MAX_PROMPTS_GENERIC" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --decoding greedy \
  --system_prompt_mode none \
  --output_file "$OUTPUT_DIR/steering/sweep_granularity.jsonl"

if [ "$RUN_MICRO_TARGETED" = "1" ]; then
  # Micro-targeted prompts (paper main): more local/personal prompts (12 prompts).
  python pipeline/6_steer.py \
    --model "$MODEL" \
    --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --roles_dir "$ROLES_DIR" \
    --metadata_file "$METADATA_FILE" \
    --layers "$TARGET_LAYER" \
    --coeffs $STEERING_COEFFS \
    --directions granularity \
    --prompts_file data/steering_prompts_micro.jsonl \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --decoding greedy \
    --system_prompt_mode none \
    --output_file "$OUTPUT_DIR/steering/sweep_granularity_micro.jsonl"
fi

python pipeline/6_steer.py \
  --model "$MODEL" \
  --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
  --vectors_dir "$OUTPUT_DIR/vectors" \
  --roles_dir "$ROLES_DIR" \
  --metadata_file "$METADATA_FILE" \
  --layers "$TARGET_LAYER" \
  --coeffs $BASELINE_COEFFS \
  --directions assistant random \
  --prompts_file data/steering_prompts.jsonl \
  --max_prompts "$MAX_PROMPTS_BASELINE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --decoding greedy \
  --system_prompt_mode none \
  --output_file "$OUTPUT_DIR/steering/sweep_baselines.jsonl"

if [ "$RUN_DECODING_SENSITIVITY" = "1" ]; then
  python pipeline/6_steer.py \
    --model "$MODEL" \
    --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
    --layers "$TARGET_LAYER" \
    --coeffs -4.0 0.0 4.0 \
    --directions granularity \
    --prompts_file data/steering_prompts.jsonl \
    --max_prompts "$MAX_PROMPTS_SAMPLED" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --decoding sample \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 50 \
    --system_prompt_mode none \
    --output_file "$OUTPUT_DIR/steering/sweep_sampled.jsonl"
fi

echo ""
echo "=== Stage 8: Steering evaluation (text metrics + optional judge) ==="
python pipeline/9_text_metrics.py \
  --responses_file "$OUTPUT_DIR/steering/sweep_granularity.jsonl" \
  --output_dir "$OUTPUT_DIR/analysis/text_metrics/granularity"
if [ "$RUN_MICRO_TARGETED" = "1" ]; then
  python pipeline/9_text_metrics.py \
    --responses_file "$OUTPUT_DIR/steering/sweep_granularity_micro.jsonl" \
    --output_dir "$OUTPUT_DIR/analysis/text_metrics/granularity_micro"
fi
python pipeline/9_text_metrics.py \
  --responses_file "$OUTPUT_DIR/steering/sweep_baselines.jsonl" \
  --output_dir "$OUTPUT_DIR/analysis/text_metrics/baselines"
if [ "$RUN_DECODING_SENSITIVITY" = "1" ]; then
  python pipeline/9_text_metrics.py \
    --responses_file "$OUTPUT_DIR/steering/sweep_sampled.jsonl" \
    --output_dir "$OUTPUT_DIR/analysis/text_metrics/sampled"
fi

if [ -n "$OPENAI_API_KEY" ]; then
  for JUDGE_MODEL in $JUDGE_MODELS; do
    JUDGE_SLUG="$(printf '%s' "$JUDGE_MODEL" | tr '/: ' '___')"
    python pipeline/8_judge_steering.py \
      --responses_file "$OUTPUT_DIR/steering/sweep_granularity.jsonl" \
      --rubric_file data/steering_judge_rubric.json \
      --output_dir "$OUTPUT_DIR/analysis/judge_scores/$JUDGE_SLUG/granularity" \
      --judge_model "$JUDGE_MODEL" \
      --temperature "$JUDGE_TEMPERATURE" \
      --batch_size "$JUDGE_BATCH_SIZE" \
      --requests_per_second "$JUDGE_RPS"

    if [ "$RUN_MICRO_TARGETED" = "1" ]; then
      python pipeline/8_judge_steering.py \
        --responses_file "$OUTPUT_DIR/steering/sweep_granularity_micro.jsonl" \
        --rubric_file data/steering_judge_rubric.json \
        --output_dir "$OUTPUT_DIR/analysis/judge_scores/$JUDGE_SLUG/granularity_micro" \
        --judge_model "$JUDGE_MODEL" \
        --temperature "$JUDGE_TEMPERATURE" \
        --batch_size "$JUDGE_BATCH_SIZE" \
        --requests_per_second "$JUDGE_RPS"
    fi

    python pipeline/8_judge_steering.py \
      --responses_file "$OUTPUT_DIR/steering/sweep_baselines.jsonl" \
      --rubric_file data/steering_judge_rubric.json \
      --output_dir "$OUTPUT_DIR/analysis/judge_scores/$JUDGE_SLUG/baselines" \
      --judge_model "$JUDGE_MODEL" \
      --temperature "$JUDGE_TEMPERATURE" \
      --batch_size "$JUDGE_BATCH_SIZE" \
      --requests_per_second "$JUDGE_RPS"
  done
else
  echo "[SKIP] Judge scoring skipped (OPENAI_API_KEY/API_KEY not set)."
fi

if [ "$RUN_SCORE_FILTERING" = "1" ]; then
  echo ""
  echo "=== Stage 9: Score filtering ablation (optional) ==="
  if [ -n "$OPENAI_API_KEY" ]; then
    python pipeline/3_judge.py \
      --responses_dir "$OUTPUT_DIR/responses" \
      --roles_dir "$ROLES_DIR" \
      --output_dir "$OUTPUT_DIR/scores" \
      --judge_model "$ROLE_JUDGE_MODEL" \
      --temperature "$JUDGE_TEMPERATURE"

    python pipeline/4_vectors.py \
      --activations_dir "$OUTPUT_DIR/activations" \
      --roles_dir "$ROLES_DIR" \
      --metadata_file "$METADATA_FILE" \
      --scores_dir "$OUTPUT_DIR/scores" \
      --output_dir "$OUTPUT_DIR/vectors_score${MIN_SCORE_STRICT}" \
      --min_score "$MIN_SCORE_STRICT" \
      --score_mode at_least \
      --min_count 10

    python pipeline/5_axis.py \
      --vectors_dir "$OUTPUT_DIR/vectors_score${MIN_SCORE_STRICT}" \
      --output_dir "$OUTPUT_DIR/axis_score${MIN_SCORE_STRICT}" \
      --micro_levels $MICRO_LEVELS \
      --macro_levels $MACRO_LEVELS \
      --target_layer "$TARGET_LAYER"

    python pipeline/4_vectors.py \
      --activations_dir "$OUTPUT_DIR/activations" \
      --roles_dir "$ROLES_DIR" \
      --metadata_file "$METADATA_FILE" \
      --scores_dir "$OUTPUT_DIR/scores" \
      --output_dir "$OUTPUT_DIR/vectors_score${MIN_SCORE_LOOSE}" \
      --min_score "$MIN_SCORE_LOOSE" \
      --score_mode at_least \
      --min_count 10

    python pipeline/5_axis.py \
      --vectors_dir "$OUTPUT_DIR/vectors_score${MIN_SCORE_LOOSE}" \
      --output_dir "$OUTPUT_DIR/axis_score${MIN_SCORE_LOOSE}" \
      --micro_levels $MICRO_LEVELS \
      --macro_levels $MACRO_LEVELS \
      --target_layer "$TARGET_LAYER"

    python pipeline/10_subgroup_analysis.py \
      --vectors_dir "$OUTPUT_DIR/vectors" \
      --activations_dir "$OUTPUT_DIR/activations" \
      --roles_dir "$ROLES_DIR" \
      --metadata_file "$METADATA_FILE" \
      --scores_dir "$OUTPUT_DIR/scores" \
      --output_dir "$OUTPUT_DIR/analysis/subgroups_scored" \
      --target_layer "$TARGET_LAYER" \
      --micro_levels $MICRO_LEVELS \
      --macro_levels $MACRO_LEVELS
  else
    echo "[SKIP] Score filtering requires OPENAI_API_KEY/API_KEY."
  fi
fi

if [ "$RUN_STEERING_ROBUSTNESS" = "1" ]; then
  echo ""
  echo "=== Stage 10: Steering robustness (optional) ==="

  ROBUST_DIR="$OUTPUT_DIR/steering_robustness"
  STEERING_DIR="$ROBUST_DIR/steering"
  TEXT_DIR="$ROBUST_DIR/analysis/text_metrics"
  JUDGE_DIR="$ROBUST_DIR/analysis/judge_scores"
  SUMMARY_DIR="$ROBUST_DIR/analysis/steering_robustness_summary"
  mkdir -p "$STEERING_DIR" "$TEXT_DIR" "$JUDGE_DIR" "$SUMMARY_DIR"

  python pipeline/6_steer.py \
    --model "$MODEL" \
    --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --roles_dir "$ROLES_DIR" \
    --metadata_file "$METADATA_FILE" \
    --layers "$TARGET_LAYER" \
    --coeffs -4.0 4.0 \
    --directions granularity \
    --prompts_file data/steering_prompts.jsonl \
    --max_new_tokens "$TRUNCATION_SHORT_TOKENS" \
    --decoding greedy \
    --system_prompt_mode none \
    --output_file "$STEERING_DIR/truncation_${TRUNCATION_SHORT_TOKENS}_granularity.jsonl"

  python pipeline/6_steer.py \
    --model "$MODEL" \
    --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --roles_dir "$ROLES_DIR" \
    --metadata_file "$METADATA_FILE" \
    --layers "$TARGET_LAYER" \
    --coeffs -4.0 4.0 \
    --directions granularity \
    --prompts_file data/steering_prompts.jsonl \
    --max_new_tokens "$TRUNCATION_LONG_TOKENS" \
    --decoding greedy \
    --system_prompt_mode none \
    --output_file "$STEERING_DIR/truncation_${TRUNCATION_LONG_TOKENS}_granularity.jsonl"

  python pipeline/6_steer.py \
    --model "$MODEL" \
    --axis_file "$OUTPUT_DIR/axis/granularity_axis.pt" \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --roles_dir "$ROLES_DIR" \
    --metadata_file "$METADATA_FILE" \
    --layers "$TARGET_LAYER" \
    --coeffs -4.0 4.0 \
    --directions granularity \
    --prompts_file data/steering_prompts_micro.jsonl \
    --max_new_tokens "$TRUNCATION_LONG_TOKENS" \
    --decoding greedy \
    --system_prompt_mode none \
    --output_file "$STEERING_DIR/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity.jsonl"

  python pipeline/9_text_metrics.py \
    --responses_file "$STEERING_DIR/truncation_${TRUNCATION_SHORT_TOKENS}_granularity.jsonl" \
    --output_dir "$TEXT_DIR/truncation_${TRUNCATION_SHORT_TOKENS}_granularity"
  python pipeline/9_text_metrics.py \
    --responses_file "$STEERING_DIR/truncation_${TRUNCATION_LONG_TOKENS}_granularity.jsonl" \
    --output_dir "$TEXT_DIR/truncation_${TRUNCATION_LONG_TOKENS}_granularity"
  python pipeline/9_text_metrics.py \
    --responses_file "$STEERING_DIR/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity.jsonl" \
    --output_dir "$TEXT_DIR/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity"

  if [ -n "$OPENAI_API_KEY" ]; then
    for JUDGE_MODEL in $JUDGE_MODELS; do
      JUDGE_SLUG="$(printf '%s' "$JUDGE_MODEL" | tr '/: ' '___')"
      python pipeline/8_judge_steering.py \
        --responses_file "$STEERING_DIR/truncation_${TRUNCATION_SHORT_TOKENS}_granularity.jsonl" \
        --rubric_file data/steering_judge_rubric.json \
        --output_dir "$JUDGE_DIR/$JUDGE_SLUG/truncation_${TRUNCATION_SHORT_TOKENS}_granularity" \
        --judge_model "$JUDGE_MODEL" \
        --temperature "$JUDGE_TEMPERATURE" \
        --batch_size "$JUDGE_BATCH_SIZE" \
        --requests_per_second "$JUDGE_RPS"

      python pipeline/8_judge_steering.py \
        --responses_file "$STEERING_DIR/truncation_${TRUNCATION_LONG_TOKENS}_granularity.jsonl" \
        --rubric_file data/steering_judge_rubric.json \
        --output_dir "$JUDGE_DIR/$JUDGE_SLUG/truncation_${TRUNCATION_LONG_TOKENS}_granularity" \
        --judge_model "$JUDGE_MODEL" \
        --temperature "$JUDGE_TEMPERATURE" \
        --batch_size "$JUDGE_BATCH_SIZE" \
        --requests_per_second "$JUDGE_RPS"

      python pipeline/8_judge_steering.py \
        --responses_file "$STEERING_DIR/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity.jsonl" \
        --rubric_file data/steering_judge_rubric.json \
        --output_dir "$JUDGE_DIR/$JUDGE_SLUG/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity" \
        --judge_model "$JUDGE_MODEL" \
        --temperature "$JUDGE_TEMPERATURE" \
        --batch_size "$JUDGE_BATCH_SIZE" \
        --requests_per_second "$JUDGE_RPS"
    done
  else
    echo "[SKIP] Judge scoring skipped (OPENAI_API_KEY/API_KEY not set)."
  fi

  PRIMARY_JUDGE_SLUG="$(printf '%s' "$PRIMARY_RISK_JUDGE_MODEL" | tr '/: ' '___')"

  python pipeline/11_steering_risk_analysis.py \
    --default_placement "$OUTPUT_DIR/analysis/representation/default_assistant_placement.json" \
    --judge_summary "main_greedy=$OUTPUT_DIR/analysis/judge_scores/$PRIMARY_JUDGE_SLUG/granularity/steering_judge_summary.json" \
    --judge_summary "truncation_${TRUNCATION_SHORT_TOKENS}=$JUDGE_DIR/$PRIMARY_JUDGE_SLUG/truncation_${TRUNCATION_SHORT_TOKENS}_granularity/steering_judge_summary.json" \
    --judge_summary "truncation_${TRUNCATION_LONG_TOKENS}=$JUDGE_DIR/$PRIMARY_JUDGE_SLUG/truncation_${TRUNCATION_LONG_TOKENS}_granularity/steering_judge_summary.json" \
    --judge_summary "micro_targeted_${TRUNCATION_LONG_TOKENS}=$JUDGE_DIR/$PRIMARY_JUDGE_SLUG/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity/steering_judge_summary.json" \
    --text_summary "main_greedy=$OUTPUT_DIR/analysis/text_metrics/granularity/steering_text_metrics_summary.json" \
    --text_summary "truncation_${TRUNCATION_SHORT_TOKENS}=$TEXT_DIR/truncation_${TRUNCATION_SHORT_TOKENS}_granularity/steering_text_metrics_summary.json" \
    --text_summary "truncation_${TRUNCATION_LONG_TOKENS}=$TEXT_DIR/truncation_${TRUNCATION_LONG_TOKENS}_granularity/steering_text_metrics_summary.json" \
    --text_summary "micro_targeted_${TRUNCATION_LONG_TOKENS}=$TEXT_DIR/micro_targeted_${TRUNCATION_LONG_TOKENS}_granularity/steering_text_metrics_summary.json" \
    --compare_pair "truncation_${TRUNCATION_SHORT_TOKENS}=truncation_${TRUNCATION_LONG_TOKENS}" \
    --compare_pair "main_greedy=truncation_${TRUNCATION_LONG_TOKENS}" \
    --compare_pair "truncation_${TRUNCATION_LONG_TOKENS}=micro_targeted_${TRUNCATION_LONG_TOKENS}" \
    --output_dir "$SUMMARY_DIR/$PRIMARY_JUDGE_SLUG"
fi

echo ""
echo "============================================================"
echo "  Done"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

