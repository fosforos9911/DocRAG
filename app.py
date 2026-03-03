import traceback
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

from main import RAGApplication, RAGConfig


def init_page():
    st.set_page_config(
        page_title="RAG 文档问答助手",
        page_icon="📚",
        layout="wide",
    )

    # 自定义一点简单但不花哨的样式
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
        }
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
        }
        .sources-badge {
            font-size: 0.8rem;
            color: #555;
        }
        .small-metric {
            font-size: 0.75rem;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_rag_app() -> RAGApplication:
    if "rag_app" not in st.session_state:
        config = RAGConfig()
        st.session_state.rag_app = RAGApplication(config=config)
    return st.session_state.rag_app


def get_chat_history() -> List[Dict[str, Any]]:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def _sync_top_k(app: RAGApplication, top_k: int) -> None:
    """
    让 sidebar 的 top_k 对检索链立即生效。
    - `answer_with_details` 会读取 `app.config.top_k`，天然生效
    - `answer_with_lcel` 使用 `app.retriever`，需要同步到 retriever.search_kwargs
    """
    app.config.top_k = top_k
    try:
        if hasattr(app, "retriever") and hasattr(app.retriever, "search_kwargs"):
            app.retriever.search_kwargs["k"] = top_k
    except Exception:
        # 兜底：不影响应用运行（最差情况下仅 LCEL top_k 不立即更新）
        pass


def _data_status(app: RAGApplication) -> Dict[str, Any]:
    project_path = getattr(app.config, "project_path", Path.cwd())
    processed_json = project_path / "processing_results.json"
    chroma_dir = Path(app.config.chroma_dir)

    chroma_has_files = chroma_dir.exists() and any(chroma_dir.iterdir())
    return {
        "processed_json_exists": processed_json.exists(),
        "processed_json_path": processed_json,
        "chroma_dir": chroma_dir,
        "chroma_has_files": chroma_has_files,
    }


def sidebar_controls(app: RAGApplication) -> Dict[str, Any]:
    st.sidebar.title("设置")
    st.sidebar.markdown("调整问答模式与检索参数。")

    mode = st.sidebar.radio(
        "回答模式",
        options=[
            "智能工具模式",
            "RAG 详细模式",
            "RAG 简洁模式",
        ],
        index=0,
    )

    with st.sidebar.expander("检索参数", expanded=True):
        top_k = st.slider(
            "检索文档数量 (top_k)",
            min_value=1,
            max_value=10,
            value=app.config.top_k,
            step=1,
        )
        _sync_top_k(app, top_k)

    show_sources = st.sidebar.checkbox("显示信息来源", value=True)
    show_stats = st.sidebar.checkbox("显示时间统计", value=True)

    if st.sidebar.button("清空对话", type="secondary"):
        st.session_state.chat_history = []
        try:
            app.conversation_history = []
        except Exception:
            pass
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("数据状态")
    status = _data_status(app)
    if status["chroma_has_files"]:
        st.sidebar.success("向量库已就绪")
    else:
        st.sidebar.warning("向量库看起来还未构建或为空")
        if not status["processed_json_exists"]:
            st.sidebar.caption(
                "未发现 `processing_results.json`。通常需要先运行 `data_preprocess.py`。"
            )
        st.sidebar.caption(
            "如果你已经生成了 JSON，接着运行构建向量库的脚本（例如 `vector_store.py` 里的入口逻辑）。"
        )

    st.sidebar.caption(
        "当前向量库目录：\n"
        f"`{status['chroma_dir']}`"
    )

    return {
        "mode": mode,
        "show_sources": show_sources,
        "show_stats": show_stats,
    }


def render_header():
    st.markdown(
        """
        <div class="chat-container">
            <h2>📚 RAG 文档问答助手</h2>
            <p style="color:#555;">
                基于本地文档向量库 + DeepSeek 模型的问答系统。建议用“RAG 详细模式”获取来源与耗时统计。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def call_rag(app: RAGApplication, question: str, mode: str) -> Dict[str, Any]:
    if mode == "智能工具模式":
        answer = app.answer_with_tools(question)
        return {
            "answer": answer,
            "sources": [],
            "retrieval_time": None,
            "generation_time": None,
            "total_time": None,
        }

    if mode == "RAG 详细模式":
        return app.answer_with_details(question)

    if mode == "RAG 简洁模式":
        answer = app.answer_with_lcel(question)
        return {
            "answer": answer,
            "sources": [],
            "retrieval_time": None,
            "generation_time": None,
            "total_time": None,
        }

    answer = app.answer_with_lcel(question)
    return {
        "answer": answer,
        "sources": [],
        "retrieval_time": None,
        "generation_time": None,
        "total_time": None,
    }


def render_chat():
    app = get_rag_app()
    controls = sidebar_controls(app)
    render_header()

    history = get_chat_history()

    with st.container():
        for item in history:
            with st.chat_message("user"):
                st.markdown(item["question"])

            with st.chat_message("assistant"):
                st.markdown(item["answer"])

                if controls["show_sources"] and item.get("sources"):
                    sources_text = ", ".join(sorted(set(item["sources"])))
                    st.markdown(
                        f'<div class="sources-badge">📎 信息来源：{sources_text}</div>',
                        unsafe_allow_html=True,
                    )

                if controls["show_stats"]:
                    rt = item.get("retrieval_time")
                    gt = item.get("generation_time")
                    tt = item.get("total_time")
                    stats = []
                    if tt is not None:
                        stats.append(f"总耗时 {tt:.2f}s")
                    if rt is not None:
                        stats.append(f"检索 {rt:.2f}s")
                    if gt is not None:
                        stats.append(f"生成 {gt:.2f}s")
                    if stats:
                        st.markdown(
                            f'<div class="small-metric">⏱ {" | ".join(stats)}</div>',
                            unsafe_allow_html=True,
                        )

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    user_input = st.chat_input("在这里输入你的问题，例如：公司的成立时间是什么？")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner("正在检索文档并生成回答…"):
                    result = call_rag(app, user_input, controls["mode"])

                answer = result.get("answer", "")
                sources = result.get("sources", [])
                retrieval_time = result.get("retrieval_time")
                generation_time = result.get("generation_time")
                total_time = result.get("total_time")

                history.append(
                    {
                        "question": user_input,
                        "answer": answer,
                        "sources": sources,
                        "retrieval_time": retrieval_time,
                        "generation_time": generation_time,
                        "total_time": total_time,
                    }
                )

                st.markdown(answer)

                if controls["show_sources"] and sources:
                    sources_text = ", ".join(sorted(set(sources)))
                    st.markdown(
                        f'<div class="sources-badge">📎 信息来源：{sources_text}</div>',
                        unsafe_allow_html=True,
                    )

                if controls["show_stats"]:
                    stats = []
                    if total_time is not None:
                        stats.append(f"总耗时 {total_time:.2f}s")
                    if retrieval_time is not None:
                        stats.append(f"检索 {retrieval_time:.2f}s")
                    if generation_time is not None:
                        stats.append(f"生成 {generation_time:.2f}s")
                    if stats:
                        st.markdown(
                            f'<div class="small-metric">⏱ {" | ".join(stats)}</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error("生成回答时出错，请查看终端日志。")
                st.caption(str(e))
                traceback.print_exc()


def main():
    init_page()
    render_chat()


if __name__ == "__main__":
    main()

