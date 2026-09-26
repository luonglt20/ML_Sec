"""Live, no-cache prompt-injection lab for the MedQA agent.

Every displayed answer is a fresh provider API response.  The app does not
use the repository's on-disk LLM cache and contains no replayed answers.

Run from the repository root:
    python -m streamlit run medqa_multiagent/ui/app.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medqa_multiagent.config import RunConfig
from medqa_multiagent.data import load_questions
from medqa_multiagent.entrypoint import SUPPORTED_VARIANTS, AnswerResult, answer_question
from medqa_multiagent.env_file import load_env_file
from medqa_multiagent.llm_client import LLMClient, create_llm_client
from medqa_multiagent.semantic_defense import RoleSeparatedGuardClient, parse_guard_response
from medqa_multiagent.struq_frontend import StruQFrontendAPIClient
from medqa_multiagent.ui.attack_demo import DemoCase, build_demo_cases


load_env_file(str(PROJECT_ROOT / ".env"))
st.set_page_config(page_title="MedQA Live Security Lab", page_icon="🛡️", layout="wide")


@dataclass(frozen=True)
class LiveComparison:
    """Results of three fresh answer calls for one case."""

    case: DemoCase
    defense_name: str
    defended_input: str
    normal: AnswerResult
    attack: AnswerResult
    defense: AnswerResult
    defense_detail: str
    structured_prompt: Optional[str] = None


def is_live_comparison(value: object) -> bool:
    """Accept an object restored from Streamlit session state after rerun.

    Streamlit reruns this script after a successful scan, redefining the
    dataclass. An instance saved before that rerun then fails ``isinstance``
    despite still containing a valid comparison. Check the required shape
    instead so the three result cards remain visible.
    """
    return all(
        hasattr(value, field)
        for field in ("case", "defense_name", "defended_input", "normal", "attack", "defense")
    )


def load_demo_cases(variant: str) -> list[DemoCase]:
    """Read only question data; no answer or API result is cached locally."""
    return build_demo_cases(load_questions(PROJECT_ROOT / "data" / "test.jsonl"), variant)


def load_run_config() -> RunConfig:
    return RunConfig.from_json_file(PROJECT_ROOT / "config_deepseek.json")


def semantic_guard(question: str, config: RunConfig) -> str:
    """Call the role-separated semantic guard API; it is never cache-wrapped."""
    response = RoleSeparatedGuardClient().complete(
        role="security_guard",
        prompt=question,
        model=config.model,
        temperature=0.0,
    )
    return parse_guard_response(question, response).question


def run_live_comparison(
    case: DemoCase,
    variant: str,
    defense_name: str,
    config: RunConfig,
    *,
    skip_defense_unless_attack_wrong: bool = False,
) -> Optional[LiveComparison]:
    """Make fresh API calls, skipping an unnecessary defense probe if requested."""
    # create_llm_client returns the raw provider client. Do not replace this
    # with build_llm_client: that function intentionally adds the disk cache.
    attack_input = case.attacked_question
    normal = answer_question(
        case.question.question, case.question.options, variant, config,
        create_llm_client(config.model),
    )
    attack = answer_question(
        attack_input, case.question.options, variant, config,
        create_llm_client(config.model),
    )
    # During discovery, a defense result cannot satisfy the requested demo
    # pattern unless the clean answer is right and the attack is wrong. Avoid
    # spending a guard/API call for every other candidate.
    if skip_defense_unless_attack_wrong and (
        normal.answer != case.question.answer or attack.answer == case.question.answer
    ):
        return None

    defended_input = attack_input
    structured_prompt: Optional[str] = None
    defense_detail = ""
    defense_client: LLMClient = create_llm_client(config.model)
    if defense_name.startswith("semantic"):
        defended_input = semantic_guard(attack_input, config)
        defense_detail = "semantic_guard made a separate API call before the fresh answer API call."
    elif defense_name.startswith("StruQ"):
        struq_client = StruQFrontendAPIClient(
            create_llm_client(config.model), [attack_input]
        )
        defense_client = struq_client
        defense_detail = (
            "StruQ-compatible frontend separated the untrusted question into a data channel; "
            "the answer still comes from a fresh DeepSeek API call."
        )
    else:
        raise ValueError(f"Unsupported defense: {defense_name}")
    return LiveComparison(
        case=case,
        defense_name=defense_name,
        defended_input=defended_input,
        normal=normal,
        attack=attack,
        defense=answer_question(
            defended_input, case.question.options, variant, config, defense_client,
        ),
        defense_detail=defense_detail,
        structured_prompt=struq_client.last_structured_prompt if defense_name.startswith("StruQ") else None,
    )


def verdict(answer: Optional[str], gold: str) -> str:
    if answer is None:
        return "Invalid output"
    return "Correct" if answer == gold else "Wrong"


def api_detail(result: AnswerResult) -> str:
    return (
        f"Fresh API call · cache_hit=False · {result.total_tokens} tokens · "
        f"{result.latency_seconds:.2f}s"
    )


def render_result(title: str, result: AnswerResult, gold: str) -> None:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.metric("Agent answer", result.answer or "invalid")
        (st.success if result.answer == gold else st.error)(verdict(result.answer, gold))
        st.caption(api_detail(result))


def is_desired_demo(comparison: LiveComparison) -> bool:
    """True only for a real Normal-correct → attack-wrong → recovered run."""
    gold = comparison.case.question.answer
    return (
        comparison.normal.answer == gold
        and comparison.attack.answer != gold
        and comparison.defense.answer == gold
    )


def clear_live_result() -> None:
    """Discard cards from a different question or defense configuration."""
    st.session_state.pop("comparison", None)
    st.session_state.pop("last_scan", None)
    st.session_state.pop("pending_candidate_index", None)


def reset_case_after_variant_change() -> None:
    """A candidate index is only meaningful within one agent variant."""
    clear_live_result()
    st.session_state["candidate_index"] = 0


# A scan finishes after the candidate selectbox has already been created for
# that run.  Apply the requested dropdown change at the beginning of the next
# rerun, before Streamlit instantiates that widget.
pending_candidate_index = st.session_state.pop("pending_candidate_index", None)
if pending_candidate_index is not None:
    st.session_state["candidate_index"] = pending_candidate_index


st.title("MedQA Live Security Lab")
st.caption("Normal, prompt injection, và defense đều gọi provider API mới. Không dùng replay hoặc LLM cache trên đĩa.")

with st.sidebar:
    st.header("Thiết lập live run")
    variant = st.selectbox(
        "Agent variant", SUPPORTED_VARIANTS, index=0,
        key="variant_selection", on_change=reset_case_after_variant_change,
    )
    defense_name = st.selectbox(
        "Defense",
        (
            "semantic_guard (DeepSeek)",
            "StruQ-compatible frontend (DeepSeek API)",
        ),
        key="defense_selection",
        on_change=clear_live_result,
    )
    if defense_name.startswith("StruQ"):
        st.caption(
            "API-only structural frontend. Không phải checkpoint StruQ Mistral được fine-tune."
        )
    else:
        st.caption("`semantic_guard` thêm một API call trước khi gửi answer request tới model.")
    st.divider()
    st.caption("Attack template: `combine` — fake completion + ignore instructions + targeted wrong answer.")

try:
    cases = load_demo_cases(variant)
except Exception as exc:
    st.error(f"Không đọc được data/test.jsonl: {exc}")
    st.stop()

if not cases:
    st.error("Không có câu hỏi demo.")
    st.stop()

st.warning(
    "Mỗi lần chạy gửi 3 API calls mới tới model (4 calls nếu dùng semantic_guard). "
    "Không có kết quả mô phỏng hoặc kết quả đọc từ `.cache/llm`."
)
st.info(
    "Các ca hiển thị được ưu tiên từ benchmark no-defense lịch sử của đúng variant: clean đúng → attacked sai. "
    "Nút tìm kiếm sẽ xác nhận lại bằng API hiện tại: Normal đúng → Attack sai → Defense đúng."
)

labels = [f"{item.question.question_id} · target sai: {item.target_answer}" for item in cases]
selected = st.selectbox(
    "Ca ứng viên từ benchmark no-defense",
    range(len(cases)),
    format_func=lambda i: labels[i],
    key="candidate_index",
    on_change=clear_live_result,
)
case = cases[selected]
question = case.question

# Be defensive about a stale browser/session state from an earlier automatic
# scan.  A manually selected case must never inherit that scan's status or
# cards, even if Streamlit did not invoke the widget callback (for example
# after a server restart).
stored_comparison = st.session_state.get("comparison")
if (
    is_live_comparison(stored_comparison)
    and stored_comparison.case.question.question_id != question.question_id
):
    clear_live_result()

left, right = st.columns((1.25, 1))
with left:
    st.subheader("Clinical question")
    st.write(question.question)
    for letter, option in question.options.items():
        st.write(f"**{letter}.** {option}")
with right:
    st.subheader("Attack payload (untrusted input)")
    st.code(case.attacked_question, language=None)
    st.caption(f"Gold evaluation label: {question.answer} · Target injection: {case.target_answer}")

run_selected_case = st.button(
    "Chạy 3 điều kiện cho ca đang chọn",
    type="primary",
    use_container_width=True,
)
# Keep the automatic-search implementation below available, but do not expose
# its trigger in the demo UI.
find_real_demo = False

try:
    config = load_run_config()
except Exception as exc:
    st.error(f"Không đọc được config_deepseek.json: {exc}")
    st.stop()

last_scan = st.session_state.get("last_scan")
if last_scan:
    kind = last_scan["kind"]
    message = last_scan["message"]
    if kind == "found":
        st.success(message)
    elif kind == "not_found":
        st.warning(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)
else:
    st.info("Chưa có lượt quét hoàn tất. Bấm nút đỏ để tìm một demo API đạt pattern.")

if run_selected_case:
    clear_live_result()
    try:
        comparison = run_live_comparison(case, variant, defense_name, config)
        st.session_state["comparison"] = comparison
        if is_desired_demo(comparison):
            message = "Ca đang chọn đã xác nhận bằng API: đúng → attack sai → khôi phục đúng."
            kind = "found"
        else:
            message = (
                "Đã chạy API cho ca đang chọn. Ca này không đạt đầy đủ pattern; "
                "ba kết quả bên dưới vẫn là kết quả API thật."
            )
            kind = "completed"
        st.session_state["last_scan"] = {"kind": kind, "message": message}
        st.rerun()
    except Exception as exc:
        st.session_state["last_scan"] = {
            "kind": "error",
            "message": f"Không thể chạy ca đang chọn bằng API: {exc}",
        }
        st.rerun()

if find_real_demo:
    # Never leave an older non-qualifying comparison visible after a fresh
    # scan. The demo either shows a proven live triplet or no triplet at all.
    st.session_state.pop("comparison", None)
    st.session_state["last_scan"] = {
        "kind": "running",
        "message": f"Đang sàng lọc tối đa {len(cases)} ca bằng API mới...",
    }
    found: Optional[LiveComparison] = None
    progress = st.progress(0, text="Đang sàng lọc bằng các API calls mới...")
    try:
        for index, candidate in enumerate(cases, start=1):
            progress.progress(
                (index - 1) / len(cases),
                text=f"Đang thử {index}/{len(cases)}: gọi Normal và Attack...",
            )
            comparison = run_live_comparison(
                candidate, variant, defense_name, config,
                skip_defense_unless_attack_wrong=True,
            )
            if comparison is not None and is_desired_demo(comparison):
                found = comparison
                break
            progress.progress(index / len(cases), text=f"Đã thử {index}/{len(cases)} ca bằng API")
        progress.empty()
        if found is None:
            st.session_state["last_scan"] = {
                "kind": "not_found",
                "message": (
                    f"Đã quét {len(cases)} ca nhưng không có ca nào thỏa toàn bộ pattern với model này. "
                    "Đây là kết quả API thật; UI không thay thế bằng replay."
                ),
            }
            st.rerun()
        else:
            st.session_state["comparison"] = found
            # Keep dropdown, clinical question, payload, and result cards on
            # the exact same case after an automatic scan selects a winner.
            st.session_state["pending_candidate_index"] = next(
                index for index, candidate in enumerate(cases)
                if candidate.question.question_id == found.case.question.question_id
            )
            st.session_state["last_scan"] = {
                "kind": "found",
                "message": f"Đã tìm thấy demo API thực: {found.case.question.question_id}.",
            }
            # Re-render from the top so the question and payload shown above
            # are the same case as the three result cards below.
            st.rerun()
    except Exception as exc:
        progress.empty()
        st.session_state["last_scan"] = {
            "kind": "error",
            "message": f"Không thể sàng lọc bằng API: {exc}",
        }
        st.rerun()

comparison = st.session_state.get("comparison")
if (
    is_live_comparison(comparison)
    and comparison.case.question.question_id == case.question.question_id
):
    if comparison.defense_name.startswith("StruQ"):
        st.success("StruQ-compatible frontend đã tách untrusted question khỏi instruction channel.")
    elif comparison.case.attacked_question != comparison.defended_input:
        st.success(f"Guard đã thay đổi input trước inference ({comparison.defense_name}).")
    else:
        st.warning("Guard không thay đổi input; kiểm tra payload hoặc semantic guard response.")
    c1, c2, c3 = st.columns(3)
    gold = comparison.case.question.answer
    with c1:
        render_result("1. Normal", comparison.normal, gold)
    with c2:
        render_result("2. Under attack", comparison.attack, gold)
    with c3:
        render_result("3. Attack + defense", comparison.defense, gold)
    if is_desired_demo(comparison):
        st.success("Pattern demo đã được xác nhận bằng API: đúng → attack sai → khôi phục đúng.")
    else:
        st.info("Kết quả này không đạt pattern demo và sẽ không được giữ sau lần quét mới.")
    with st.expander("Xem input sau khi guard xử lý"):
        st.code(comparison.defended_input, language=None)
    if comparison.structured_prompt:
        with st.expander("Xem StruQ-compatible structured prompt gửi tới API"):
            st.code(comparison.structured_prompt, language=None)
            st.caption(comparison.defense_detail)
