"""
Result Combiner — merges multi-step execution outputs into one DomainOutput.
"""

from __future__ import annotations

from shared.models import DomainOutput, ExecutionPlan, ExecutionStep


class ResultCombiner:
    """Combines DAG/sequential outputs into a stable response envelope."""

    def combine(
        self,
        plan: ExecutionPlan,
        steps_by_id: dict[int, ExecutionStep],
        outputs: dict[int, DomainOutput],
    ) -> DomainOutput:
        if not outputs:
            return DomainOutput(
                status="failure",
                result={},
                explanation="No task outputs were produced.",
                confidence=0.0,
                metadata={"error": "empty_execution"},
            )

        ordered_ids = sorted(outputs.keys())
        last_output = outputs[ordered_ids[-1]]

        if len(outputs) == 1 and plan.combine_mode == "last":
            return last_output

        parts: dict[str, dict] = {}
        required_failures: list[str] = []
        optional_failures: list[str] = []
        clarifications: list[str] = []

        for step_id in ordered_ids:
            step = steps_by_id[step_id]
            out = outputs[step_id]
            key = step.output_key or f"step_{step_id}"

            parts[key] = {
                "step_id": step_id,
                "domain": step.domain,
                "capability": step.capability,
                "required": step.required,
                "status": out.status,
                "explanation": out.explanation,
                "result": out.result,
                "metadata": out.metadata,
            }

            if out.status == "clarification":
                clarifications.append(out.explanation)
                if step.required:
                    required_failures.append(f"step_{step_id}")
                else:
                    optional_failures.append(f"step_{step_id}")
            elif out.status == "failure":
                if step.required:
                    required_failures.append(f"step_{step_id}")
                else:
                    optional_failures.append(f"step_{step_id}")

        if required_failures:
            status = "clarification" if clarifications else "failure"
            explanation = (
                clarifications[0]
                if clarifications
                else f"Execution failed in required task(s): {', '.join(required_failures)}."
            )
            confidence = 0.0 if status == "failure" else 0.8
        else:
            status = "success"
            confidence = 1.0
            explanation = f"Executed {len(outputs)} task(s)."
            if optional_failures:
                explanation += f" Optional failures: {', '.join(optional_failures)}."

        result = {
            "combined": last_output.result if plan.combine_mode in ("last", "report") else {},
            "steps": parts,
            "required_failures": required_failures,
            "optional_failures": optional_failures,
        }

        metadata = {
            "execution_mode": plan.execution_mode,
            "combine_mode": plan.combine_mode,
            "steps_executed": len(outputs),
        }

        return DomainOutput(
            status=status,
            result=result,
            explanation=explanation,
            confidence=confidence,
            metadata=metadata,
        )
