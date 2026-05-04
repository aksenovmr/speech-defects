import uuid
import os
import streamlit as st
import requests
import pandas as pd
from src.utils.config import load_yaml_config

cfg = load_yaml_config("configs/inference.yaml")
API_URL = os.getenv("API_URL", cfg["streamlit"]["api_url"])


def load_health():
    response = requests.get(f"{API_URL}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_history(limit: int = 100, session_id: str | None = None):
    params = {"limit": limit}
    if session_id:
        params["session_id"] = session_id

    response = requests.get(f"{API_URL}/history", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def render_status(status: str):
    if status == "good":
        st.success("Статус: good")
    elif status == "warning":
        st.warning("Статус: warning")
    else:
        st.error("Статус: bad")


def render_history_table(items):
    if not items:
        st.info("В этой сессии пока нет запросов.")
        return

    rows = []
    for item in items:
        rows.append(
            {
                "ID": item.get("id"),
                "Время": item.get("created_at"),
                "Файл": item.get("filename"),
                "Проверяемые звуки": ", ".join(item.get("expected_sounds", [])),
                "p_bad": round(float(item.get("p_bad", 0.0)), 4),
                "Статус": item.get("status"),
                "Проблемные звуки": ", ".join(item.get("flagged_sounds", [])),
                "Версия модели": item.get("model_version", "unknown"),
            }
        )

    df_hist = pd.DataFrame(rows)
    st.dataframe(df_hist, use_container_width=True)

    csv_data = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Скачать историю в CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    st.set_page_config(
        page_title="Детекция дефектов речи",
        layout="wide",
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "selected_sounds" not in st.session_state:
        st.session_state.selected_sounds = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    st.title("Детекция дефектов речи")
    st.caption(
        "Загрузите аудиофайл и выберите звуки, которые проверяются в скороговорке. "
        "Анализ выполняется через FastAPI"
    )

    sounds = []
    health_data = None

    with st.sidebar:
        st.header("Backend")

        try:
            health_data = load_health()
            sounds = health_data.get("sounds", [])

            st.success("API доступен")
            st.write(f"**URL:** {API_URL}")
            st.write(f"**device:** {health_data.get('device')}")
            st.write(f"**num_sounds:** {health_data.get('num_sounds')}")
        except Exception as e:
            st.error(f"API недоступен: {e}")
            st.stop()

        st.divider()
        st.header("Подсказки")
        st.write("Поддерживаются аудиофайлы: WAV, MP3, M4A, OGG.")
        st.write("Выберите хотя бы один звук из тех, что действительно проверяются скороговоркой.")

    tab1, tab2, tab3 = st.tabs(["Новый анализ", "Мои запросы", "Batch анализ"])

    with tab1:
        col_left, col_right = st.columns([1.1, 1.2])

        with col_left:
            st.subheader("1. Загрузка аудио")

            uploaded_file = st.file_uploader(
                "Выберите аудиофайл",
                type=["wav", "mp3", "m4a", "ogg"],
                help="Можно загружать локальные аудиофайлы",
            )

            if uploaded_file is not None:
                st.audio(uploaded_file)

                file_info = {
                    "Имя файла": uploaded_file.name,
                    "Размер (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
                    "MIME type": uploaded_file.type or "unknown",
                }
                with st.expander("Информация о файле"):
                    for k, v in file_info.items():
                        st.write(f"**{k}:** {v}")

            st.subheader("2. Выбор проверяемых звуков")
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                if st.button("Выбрать все звуки", use_container_width=True):
                    st.session_state.selected_sounds = sounds.copy()

            with action_col2:
                if st.button("Очистить выбор", use_container_width=True):
                    st.session_state.selected_sounds = []

            selected_sounds = []
            num_cols = 6
            cols = st.columns(num_cols)

            for i, sound in enumerate(sounds):
                checked = sound in st.session_state.selected_sounds
                if cols[i % num_cols].checkbox(sound, value=checked, key=f"sound_{sound}"):
                    selected_sounds.append(sound)

            st.session_state.selected_sounds = selected_sounds

            if selected_sounds:
                st.write("**Выбрано:**", ", ".join(selected_sounds))
            else:
                st.info("Пока не выбран ни один звук.")

            analyze_clicked = st.button("Анализировать", type="primary", use_container_width=True)

        with col_right:
            st.subheader("3. Результат")

            if not uploaded_file:
                st.info("Загрузите аудиофайл, чтобы увидеть результат анализа.")
            elif not selected_sounds:
                st.info("Выберите проверяемые звуки.")
            elif not analyze_clicked and st.session_state.last_result is None:
                st.info("Нажмите кнопку «Анализировать».")
            else:
                if analyze_clicked:
                    try:
                        file_bytes = uploaded_file.getvalue()

                        files = {
                            "file": (
                                uploaded_file.name,
                                file_bytes,
                                uploaded_file.type or "application/octet-stream",
                            )
                        }
                        data = [("expected_sounds", sound) for sound in selected_sounds]
                        data.append(("session_id", st.session_state.session_id))

                        with st.spinner("Выполняется анализ..."):
                            response = requests.post(
                                f"{API_URL}/predict",
                                files=files,
                                data=data,
                                timeout=120,
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.session_state.last_result = result

                    except requests.HTTPError:
                        try:
                            detail = response.json()
                        except Exception:
                            detail = response.text
                        st.error(f"Ошибка API: {detail}")
                        return
                    except Exception as e:
                        st.error(f"Не удалось выполнить запрос к API: {e}")
                        return

                result = st.session_state.get("last_result")
                if not result:
                    st.warning("Нет результата анализа.")
                    return

                status = result.get("status", "unknown")
                p_bad = result.get("p_bad", 0.0)
                message = result.get("message", "")
                expected_sounds = result.get("expected_sounds", [])
                flagged_sounds = result.get("flagged_sounds", [])
                checked_sounds = result.get("checked_sounds", [])
                normal_sounds = result.get("normal_sounds", [])
                possible_issue_sounds = result.get("possible_issue_sounds", [])
                clear_issue_sounds = result.get("clear_issue_sounds", [])
                sound_items = result.get("sound_items", [])

                render_status(status)

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Вероятность дефекта", f"{p_bad:.3f}")
                with metric_col2:
                    st.metric("Число проверяемых звуков", len(expected_sounds))

                st.progress(min(max(float(p_bad), 0.0), 1.0))
                st.write(message)

                st.write("**Проверяемые звуки:**", ", ".join(expected_sounds) if expected_sounds else "—")
                st.write("**Подозрительные звуки:**", ", ".join(flagged_sounds) if flagged_sounds else "не обнаружены")

                group_col1, group_col2, group_col3 = st.columns(3)

                with group_col1:
                    st.write("**Норма**")
                    if normal_sounds:
                        for s in normal_sounds:
                            st.success(s)
                    else:
                        st.write("—")

                with group_col2:
                    st.write("**Возможные проблемы**")
                    if possible_issue_sounds:
                        for s in possible_issue_sounds:
                            st.warning(s)
                    else:
                        st.write("—")

                with group_col3:
                    st.write("**Явные проблемы**")
                    if clear_issue_sounds:
                        for s in clear_issue_sounds:
                            st.error(s)
                    else:
                        st.write("—")

                if sound_items:
                    st.subheader("Анализ по звукам")

                    df = pd.DataFrame(sound_items)
                    df = df.rename(
                        columns={
                            "sound": "Звук",
                            "prob": "Вероятность",
                            "level": "Уровень",
                        }
                    )
                    df["Вероятность"] = df["Вероятность"].map(lambda x: round(float(x), 4))
                    st.dataframe(df, use_container_width=True)

                with st.expander("Технические детали"):
                    st.write("**Имя файла:**", result.get("file_name", uploaded_file.name))
                    st.write("**checked_sounds:**", checked_sounds)
                    st.write("**normal_sounds:**", normal_sounds)
                    st.write("**possible_issue_sounds:**", possible_issue_sounds)
                    st.write("**clear_issue_sounds:**", clear_issue_sounds)
                    st.write("**session_id:**", st.session_state.session_id)
                    st.json(result)

    with tab2:
        st.subheader("Мои запросы")

        hist_col1, hist_col2 = st.columns([1, 3])
        with hist_col1:
            limit = st.slider("Сколько записей показать", min_value=10, max_value=200, value=50, step=10)
        with hist_col2:
            st.write("")
            st.button("Обновить историю", use_container_width=False)

        try:
            history_payload = fetch_history(
                limit=limit,
                session_id=st.session_state.session_id,
            )
            items = history_payload.get("items", [])
            render_history_table(items)
        except Exception as e:
            st.error(f"Не удалось загрузить историю: {e}")


    with tab3:
        st.subheader("Batch анализ")

        uploaded_files = st.file_uploader(
            "Загрузите несколько аудиофайлов",
            type=["wav", "mp3", "m4a", "ogg"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"Загружено файлов: {len(uploaded_files)}")

        selected_sounds = st.session_state.selected_sounds

        if not selected_sounds:
            st.warning("Сначала выберите звуки во вкладке 'Новый анализ'")

        run_batch = st.button("Запустить batch анализ", type="primary")

        if run_batch and uploaded_files and selected_sounds:
            try:
                files = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded_files
                ]

                data = [("expected_sounds", s) for s in selected_sounds]
                data.append(("session_id", st.session_state.session_id))

                with st.spinner("Обрабатываем файлы..."):
                    response = requests.post(
                        f"{API_URL}/predict_batch",
                        files=files,
                        data=data,
                        timeout=300,
                    )
                    response.raise_for_status()
                    result = response.json()

                rows = []
                for item in result["results"]:
                    rows.append({
                        "Файл": item.get("file_name"),
                        "p_bad": round(item.get("p_bad", 0), 4) if "p_bad" in item else None,
                        "Статус": item.get("status"),
                        "Проблемные звуки": ", ".join(item.get("flagged_sounds", [])),
                        "Ошибка": item.get("error"),
                    })

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать CSV",
                    data=csv,
                    file_name="batch_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Ошибка batch анализа: {e}")


if __name__ == "__main__":
    main()