import os, shutil, tempfile, subprocess
from datetime import timedelta
from pathlib import Path

import streamlit as st

# ================== FFmpeg helpers ==================
def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode(errors="ignore")
    if p.returncode != 0:
        raise RuntimeError(out)
    return out

def ensure_ffmpeg():
    try:
        run(["ffmpeg", "-version"])  # will raise if not found
        run(["ffprobe", "-version"])  # will raise if not found
        return True, "FFmpeg OK"
    except Exception as ex:
        return False, f"FFmpeg/ffprobe не найдены или не запускаются: {ex}\n\nУстановите FFmpeg: https://ffmpeg.org/download.html"


def hhmmss(seconds):
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=round(seconds, 3))
    s = str(td)
    if "." not in s:
        s += ".000"
    if s.count(":") == 1:
        s = "0:" + s
    if s.count(":") == 2 and len(s.split(":")[0]) == 1:
        s = "0" + s
    parts = s.split(".")
    ms = parts[1][:3].ljust(3, "0")
    h, m, sec = parts[0].split(":")
    return "%02d:%02d:%02d.%s" % (int(h), int(m), int(float(sec)), ms)


def probe_duration(path):
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    return float(out.strip())


def probe_fps(path):
    out = (
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
        )
        .strip()
        .replace("\n", "")
    )
    try:
        if "/" in out:
            a, b = out.split("/")
            a = float(a)
            b = float(b) if float(b) != 0 else 1.0
            return a / b
        return float(out)
    except Exception:
        return 30.0


def cut_segment(inp, outp, ss, to, mode, vcodec, acodec, crf=None, abitrate=None):
    ss_s, to_s = hhmmss(ss), hhmmss(to)
    dur = max(0.0, to - ss)
    if mode == "fast":
        cmd = ["ffmpeg", "-y", "-ss", ss_s, "-i", inp, "-t", f"{dur:.3f}", "-c", "copy", outp]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            ss_s,
            "-to",
            to_s,
            "-i",
            inp,
            "-c:v",
            vcodec,
            "-c:a",
            acodec,
        ]
        if crf is not None and vcodec in {"libx264", "libx265"}:
            cmd += ["-crf", str(crf)]
        if abitrate:
            cmd += ["-b:a", str(abitrate)]
        cmd += [outp]
    run(cmd)


def extract_audio(inp, outp, ss, to, mode, acodec, abitrate=None):
    ss_s, to_s = hhmmss(ss), hhmmss(to)
    dur = max(0.0, to - ss)
    if mode == "fast":
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            ss_s,
            "-i",
            inp,
            "-t",
            f"{dur:.3f}",
            "-vn",
            "-c:a",
            "copy",
            outp,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            ss_s,
            "-to",
            to_s,
            "-i",
            inp,
            "-vn",
            "-c:a",
            acodec,
        ]
        if abitrate:
            cmd += ["-b:a", str(abitrate)]
        cmd += [outp]
    run(cmd)


def extract_video_muted(inp, outp, ss, to, mode, vcodec, crf=None):
    ss_s, to_s = hhmmss(ss), hhmmss(to)
    dur = max(0.0, to - ss)
    if mode == "fast":
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            ss_s,
            "-i",
            inp,
            "-t",
            f"{dur:.3f}",
            "-an",
            "-c:v",
            "copy",
            outp,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            ss_s,
            "-to",
            to_s,
            "-i",
            inp,
            "-an",
            "-c:v",
            vcodec,
        ]
        if crf is not None and vcodec in {"libx264", "libx265"}:
            cmd += ["-crf", str(crf)]
        cmd += [outp]
    run(cmd)


def mux_av(v_path, a_path, outp, vcodec="libx264", acodec="aac"):
    if a_path and Path(a_path).exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(v_path),
            "-i",
            str(a_path),
            "-c:v",
            vcodec,
            "-c:a",
            acodec,
            "-shortest",
            str(outp),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(v_path),
            "-c:v",
            vcodec,
            "-c:a",
            acodec,
            str(outp),
        ]
    run(cmd)


def smooth_merge_two_clips(c1, c2, overlap, outp, vcodec="libx264", acodec="aac"):
    d1 = probe_duration(str(c1))
    if overlap <= 0 or overlap >= d1:
        txt = Path(outp).with_suffix(".txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(f"file '{c1.as_posix()}'\n")
            f.write(f"file '{c2.as_posix()}'\n")
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(txt),
                "-c:v",
                vcodec,
                "-c:a",
                acodec,
                str(outp),
            ]
        )
        try:
            txt.unlink()
        except Exception:
            pass
        return
    offset = max(0.0, d1 - overlap)
    # normalize audio before acrossfade for stability
    filt = (
        f"[0:v][1:v]xfade=transition=fade:duration={overlap}:offset={offset},format=yuv420p[v];"
        f"[0:a]aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=44100[0a];"
        f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=44100[1a];"
        f"[0a][1a]acrossfade=d={overlap}[a]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(c1),
        "-i",
        str(c2),
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        vcodec,
        "-c:a",
        acodec,
        str(outp),
    ]
    run(cmd)


def smooth_merge_sequence(clips, overlap, outp, vcodec="libx264", acodec="aac"):
    if len(clips) == 1:
        shutil.copyfile(clips[0], outp)
        return
    wd = Path(outp).parent
    cur = clips[0]
    for i in range(1, len(clips)):
        nx = clips[i]
        tmp = wd / (f"_xfade_step_{i}.mp4")
        smooth_merge_two_clips(cur, nx, overlap, tmp, vcodec, acodec)
        if cur != clips[0] and Path(cur).exists():
            try:
                Path(cur).unlink()
            except Exception:
                pass
        cur = tmp
    if Path(outp).exists():
        try:
            Path(outp).unlink()
        except Exception:
            pass
    Path(cur).rename(outp)


# ================== Разметка сегментов ==================
def build_segments(total, seg_len, overlap, start_offset=0.0, end_limit=None):
    if end_limit is None or end_limit > total:
        end_limit = total
    segs = []
    start = max(0.0, start_offset)
    step = max(0.01, seg_len - overlap)
    while start < end_limit - 1e-6:
        end = min(start + seg_len, end_limit)
        segs.append((start, end))
        start += step
    return segs


def parse_timecode(tc, fps):
    tc = tc.strip()
    if not tc:
        raise ValueError("empty timecode")
    if tc.count(":") == 3 and all(p.isdigit() for p in tc.replace(":", " ").split()):
        h, m, s, f = tc.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / max(fps, 1e-6)
    if tc.count(":") == 2:
        h, m, s = tc.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    if tc.count(":") == 1:
        m, s = tc.split(":")
        return int(m) * 60 + float(s)
    return float(tc)


def parse_ranges(text, fps):
    segs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for sep in ["—", "–", " to ", "→", "=>"]:
            line = line.replace(sep, "-")
        if "-" not in line:
            raise ValueError(f"Нет '-' в строке: {raw}")
        a, b = [p.strip() for p in line.split("-", 1)]
        s = parse_timecode(a, fps)
        e = parse_timecode(b, fps)
        if e <= s:
            raise ValueError(f"Конец <= старт: {raw}")
        segs.append((s, e))
    return segs


# ================== UI ==================
st.set_page_config(page_title="Video Slicer", layout="centered")
st.title("🎬 Video Slicer (v2)")

ok_ffmpeg, ffmsg = ensure_ffmpeg()
if not ok_ffmpeg:
    st.error(ffmsg)
    st.stop()

uploaded = st.file_uploader("Загрузите видеофайл", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"]) 

with st.expander("⚙️ Глобальные опции перекодирования (актуальны в режиме 'accurate')", expanded=False):
    colx, coly = st.columns(2)
    with colx:
        crf = st.slider("CRF (качество видео, меньше = лучше)", 14, 35, 22)
        vcodec_default = "libx264"
        vcodec = st.text_input("Видеокодек", vcodec_default)
    with coly:
        abitrate = st.text_input("Битрейт аудио (напр. 192k)", "192k")
        acodec = st.text_input("Аудиокодек", "aac")


tabs = st.tabs(["Автонарезка", "Ручные тайминги"]) 

with tabs[0]:
    colA, colB = st.columns(2)
    with colA:
        seg_len = st.number_input("Длина сегмента (сек)", 1.0, 600.0, 28.0, 1.0)
        overlap = st.number_input("Перекрытие (сек)", 0.0, 30.0, 2.0, 0.5)
        start_offset = st.number_input("Начать с (сек)", 0.0, None, 0.0, 0.5)
    with colB:
        end_custom = st.text_input("Остановиться на (сек), опционально", "")
        mode = st.radio("Режим нарезки", ["accurate (перекодировать)", "fast (без перекодирования)"], index=0)
        file_prefix = st.text_input("Префикс имён файлов", "clip")
    colC, colD = st.columns(2)
    with colC:
        make_audio = st.checkbox("Экспортировать аудио", True)
        audio_ext = st.selectbox("Формат аудио", ["mp3", "m4a", "aac", "wav"], index=0)
    with colD:
        make_video = st.checkbox("Экспортировать видео", True)
        video_ext = st.selectbox("Формат видео", ["mp4", "mkv", "mov", "webm"], index=0)
    smooth_join_auto = st.checkbox("Склеить все (crossfade)", True)
    run_auto = st.button("Запустить (авто)")

with tabs[1]:
    st.markdown("Каждая строка: `СТАРТ - КОНЕЦ`. Поддержка `HH:MM:SS:FF`, `HH:MM:SS.mmm`, `MM:SS`, или секунды.")
    seg_text = st.text_area("Сегменты:", height=160, value="00:00:00:00 - 00:00:28:00\n00:00:26:00 - 00:00:55:00")
    crossfade_manual = st.number_input("Crossfade (сек)", 0.0, 30.0, 2.0, 0.5)
    mode_m = st.radio("Режим обрезки", ["accurate (перекодировать)", "fast (без перекодирования)"], index=0, key="mode_m")
    file_prefix_m = st.text_input("Префикс имён файлов (ручные)", "clip_m")
    make_audio_m = st.checkbox("Экспортировать аудио (ручные)", True, key="make_audio_m")
    make_video_m = st.checkbox("Экспортировать видео (ручные)", True, key="make_video_m")
    video_ext_m = st.selectbox("Формат видео (ручные)", ["mp4", "mkv", "mov", "webm"], index=0, key="video_ext_m")
    audio_ext_m = st.selectbox("Формат аудио (ручные)", ["mp3", "m4a", "aac", "wav"], index=0, key="audio_ext_m")
    run_manual = st.button("Запустить (ручные)")

st.markdown("---")

if uploaded and (run_auto or run_manual):
    workdir = Path(tempfile.mkdtemp(prefix="slicer_"))
    in_path = workdir / uploaded.name
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        total = probe_duration(str(in_path))
        fps = probe_fps(str(in_path))
    except Exception as ex:
        st.error(f"Не удалось прочитать метаданные файла: {ex}")
        st.stop()

    st.success(f"Файл: {in_path}")
    st.info("Длительность: %.3f сек (%s), FPS≈%.3f" % (total, hhmmss(total), fps))

    if run_auto:
        mode_use = "accurate" if mode.startswith("accurate") else "fast"
        make_a, make_v = make_audio, make_video
        aext, vext = audio_ext, video_ext
        file_prefix_use = file_prefix
        end_limit = None
        if end_custom.strip():
            try:
                end_limit = float(end_custom)
            except Exception:
                st.warning("Не распознал 'Остановиться на' — игнор.")
        segs = build_segments(total, seg_len, overlap, start_offset=start_offset, end_limit=end_limit)
        do_smooth_join = smooth_join_auto
        overlap_for_join = overlap
    else:
        mode_use = "accurate" if mode_m.startswith("accurate") else "fast"
        make_a, make_v = make_audio_m, make_video_m
        aext, vext = audio_ext_m, video_ext_m
        file_prefix_use = file_prefix_m
        overlap_for_join = crossfade_manual
        try:
            segs = parse_ranges(seg_text, fps)
        except Exception as ex:
            st.error(f"Ошибка парсинга: {ex}")
            segs = []
        do_smooth_join = True

    if not segs:
        st.warning("Сегменты пустые.")
        st.stop()

    st.write(f"Будет создано сегментов: **{len(segs)}**")

    # Показать таблицу сегментов
    try:
        import pandas as pd
        df = pd.DataFrame(
            {
                "#": list(range(1, len(segs) + 1)),
                "start": [round(s, 3) for s, _ in segs],
                "end": [round(e, 3) for _, e in segs],
                "duration": [round(e - s, 3) for s, e in segs],
                "start_tc": [hhmmss(s) for s, _ in segs],
                "end_tc": [hhmmss(e) for _, e in segs],
            }
        )
        st.dataframe(df, use_container_width=True)
    except Exception:
        pass

    outdir = workdir / "out"
    outdir.mkdir(exist_ok=True)

    progress = st.progress(0, text="Экспорт сегментов…")

    # 1) Экспорт локальных файлов сегментов
    for i, (s, e) in enumerate(segs, start=1):
        if make_v:
            v_out = outdir / (f"{file_prefix_use}_video_{i}.{vext}")
            try:
                cut_segment(
                    str(in_path),
                    str(v_out),
                    s,
                    e,
                    mode_use,
                    vcodec,
                    acodec,
                    crf=crf if mode_use == "accurate" else None,
                    abitrate=abitrate if mode_use == "accurate" else None,
                )
            except Exception as ex:
                st.error(f"Видео {i}: {ex}")
        if make_a:
            a_out = outdir / (f"{file_prefix_use}_audio_{i}.{aext}")
            try:
                extract_audio(
                    str(in_path),
                    str(a_out),
                    s,
                    e,
                    mode_use,
                    acodec,
                    abitrate=abitrate if mode_use == "accurate" else None,
                )
            except Exception as ex:
                st.error(f"Аудио {i}: {ex}")
        progress.progress(i / len(segs), text=f"Готово {i}/{len(segs)}…")

    progress.empty()

    # 2) Собрать A+V из локальных (если нужно)
    av_clips = []
    av_dir = outdir / "av_clips"
    av_dir.mkdir(exist_ok=True)
    for i, (s, e) in enumerate(segs, start=1):
        v_p = outdir / (f"{file_prefix_use}_video_{i}.{vext}")
        a_p = outdir / (f"{file_prefix_use}_audio_{i}.{aext}")
        if not v_p.exists() and not a_p.exists():
            continue
        av_p = av_dir / (f"clip_{i:03d}.mp4")
        try:
            mux_av(v_p if v_p.exists() else a_p, a_p if v_p.exists() else None, av_p)
        except Exception as ex:
            st.error(f"A+V {i}: {ex}")
        else:
            av_clips.append(av_p)

    # 3) Склейка (по желанию)
    if do_smooth_join and len(av_clips) >= 1:
        final_path = outdir / "final_smooth.mp4"
        try:
            st.info("Склейка (crossfade)…")
            smooth_merge_sequence(av_clips, overlap_for_join, final_path)
            st.success(f"Готово: {final_path.name}")
            with open(final_path, "rb") as f:
                st.download_button(
                    "⬇️ Скачать финальный ролик", f.read(), file_name=final_path.name, mime="video/mp4"
                )
        except Exception as ex:
            st.error(f"Склейка не удалась: {ex}")

    # 4) Пофайловые скачивания
    st.markdown("### 📥 Скачивания по файлам")
    for p in sorted(outdir.glob("*")):
        if p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".mp3", ".m4a", ".aac", ".wav"}:
            with open(p, "rb") as f:
                st.download_button(f"⬇️ {p.name}", f.read(), file_name=p.name)

    # 5) ZIP со всеми файлами
    created = []
    for i in range(1, len(segs) + 1):
        v_p = outdir / (f"{file_prefix_use}_video_{i}.{vext}")
        a_p = outdir / (f"{file_prefix_use}_audio_{i}.{aext}")
        if v_p.exists():
            created.append(v_p)
        if a_p.exists():
            created.append(a_p)
    if created:
        zip_path = outdir / "segments.zip"
        with tempfile.TemporaryDirectory() as tmpz:
            tmpdir = Path(tmpz)
            copy_root = tmpdir / "results"
            shutil.copytree(outdir, copy_root)
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", tmpdir, "results")
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Скачать ZIP (все файлы)", f.read(), file_name=zip_path.name, mime="application/zip"
            )

# ----
# Запуск локально:
#   pip install -r requirements.txt
#   streamlit run video_slicer_v2.py
# На Hugging Face Spaces используйте тип Docker Space и приложите этот Dockerfile.
