def interpret_result(
    p_bad,
    checked_sounds,
    thr_bad_good=0.2,
    thr_bad_bad=0.45,
    thr_sound_possible=0.25,
    thr_sound_clear=0.45,
):
    if p_bad < thr_bad_good:
        status = "good"
    elif p_bad < thr_bad_bad:
        status = "warning"
    else:
        status = "bad"

    sound_items = []
    for s, p in checked_sounds:
        if p < thr_sound_possible:
            level = "normal"
        elif p < thr_sound_clear:
            level = "possible_issue"
        else:
            level = "clear_issue"

        sound_items.append({
            "sound": s,
            "prob": float(p),
            "level": level,
        })

    normal = [x["sound"] for x in sound_items if x["level"] == "normal"]
    possible = [x["sound"] for x in sound_items if x["level"] == "possible_issue"]
    clear = [x["sound"] for x in sound_items if x["level"] == "clear_issue"]

    if status == "good":
        if clear or possible:
            message = (
                "В целом произношение выглядит хорошим, "
                f"но есть звуки, на которые стоит обратить внимание: {', '.join(clear + possible)}."
            )
        else:
            message = "Серьёзных признаков нарушений не обнаружено."
    elif status == "warning":
        if clear:
            message = (
                "Есть признаки возможных нарушений произношения. "
                f"Наиболее вероятные проблемные звуки: {', '.join(clear)}."
            )
        elif possible:
            message = (
                "Есть слабые признаки возможных нарушений. "
                f"Стоит обратить внимание на звуки: {', '.join(possible)}."
            )
        else:
            message = "Есть слабые признаки возможных нарушений произношения."
    else:
        if clear:
            message = (
                "Обнаружены выраженные признаки нарушений произношения. "
                f"Наиболее вероятные проблемные звуки: {', '.join(clear)}."
            )
        elif possible:
            message = (
                "Обнаружены признаки нарушений произношения. "
                f"Возможные проблемные звуки: {', '.join(possible)}."
            )
        else:
            message = "Обнаружены признаки нарушений произношения."

    return {
        "status": status,
        "message": message,
        "sound_items": sound_items,
        "normal_sounds": normal,
        "possible_issue_sounds": possible,
        "clear_issue_sounds": clear,
    }