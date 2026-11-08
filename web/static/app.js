const STATUS = {
  warmup: "Снимаю эталон…",
  ready: "Эталон снят. Можно стрелять.",
  ok: "Анализ идёт",
  "skip-shift": "Кадр сдвинулся, этот пропуск",
  "skip-motion": "Слишком много движения",
  "skip-noise": "Слишком много пятен",
  "need-calib": "Отметьте центр и край зоны 1",
  paused: "Нажмите «Старт»",
  "bad-frame": "Кадр не прочитался",
  "bad-calib": "Калибровка не сходится с кадром",
}

const LIVE = {
  ok: "Жду выстрел",
  warmup: "Снимаю эталон, не двигайте камеру",
  ready: "Эталон снят. Можно стрелять",
  "skip-shift": "Кадр сдвинулся: камера или лист качнулись",
  "skip-motion": "Слишком много движения в кадре",
  "skip-noise": "Слишком много пятен: поднимите «Порог»",
}
const FRAME_GAP_MS = 110
const ZOOM_CROP = 2.5
const CROP_OUT = 640
const RECORD_FPS = 12

const els = {}
for (const id of [
  "history", "stat-count", "stat-sum", "stat-avg", "stat-dev", "user-name", "btn-logout",
  "placeholder", "video", "preview", "view", "target", "status", "note", "btn-copy-note", "chk-carry",
  "sheet", "group", "session-no", "step-label", "btn-step-prev", "btn-step-next", "hit-log",
  "sens-val", "sens-dec", "sens-inc", "size-val", "size-dec", "size-inc",
  "download-link", "btn-center", "btn-auto", "btn-reference",
  "btn-start", "btn-new", "btn-undo", "btn-demo", "btn-camera", "ip-url", "btn-ip", "btn-save-video",
  "record-canvas", "banner", "banner-title", "banner-sub", "bar-score", "bar-tight", "overlay", "btn-zoom",
  "btn-settings", "settings", "settings-close", "set-record", "set-server", "set-zoom", "set-label",
  "set-label-val", "set-label-sample",
]) {
  els[id] = document.getElementById(id)
}

let targets = []
let state = null
let ws = null
let mode = "browser"
let running = false
let reviewIndex = null
let snaps = {}
let calibStep = null
let pendingCenter = null
let mousePoint = null
let overlayQueued = false
let demoCalib = null
let settings = { record: true, server_video: false, zoom: true, hit_label_px: 14 }
let cameraStream = null
let previewUrl = null
let recorder = null
let chunks = []
let parts = []
let downloadUrl = null
let sendTimer = null
let sheetGeom = null
let inFlight = false
let inFlightAt = 0
let tickTimes = []
let quietUntil = 0
let reconnectTimer = null

function $(id) { return els[id] }

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (typeof options.body === "string" && !headers["Content-Type"]) headers["Content-Type"] = "application/json"
  const res = await fetch(path, { ...options, headers })
  if (res.status === 401) {
    location.href = "/login"
    throw new Error("auth")
  }
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || "Ошибка")
  return data
}

function setStatus(text) {
  $("status").textContent = text
}

let bannerTimer = null

function showBanner(kind, title, sub, ms = 2200) {
  const banner = $("banner")
  banner.className = `banner ${kind}`
  $("banner-title").textContent = title
  $("banner-sub").textContent = sub
  banner.hidden = false
  void banner.offsetWidth
  banner.style.animation = "none"
  void banner.offsetWidth
  banner.style.animation = ""
  $("view").classList.toggle("wasted", kind === "miss")
  clearTimeout(bannerTimer)
  bannerTimer = setTimeout(() => {
    banner.hidden = true
    $("view").classList.remove("wasted")
  }, ms)
}

function bannerForHit(hit, earlier) {
  const sub = `${scoreLabel(hit.score)} · отклонение ${Math.round(hit.deviation)}%`
  if (hit.score === "0") return ["miss", "Мимо", `отклонение ${Math.round(hit.deviation)}%`]
  if (earlier.length && hit.deviation + 0.5 < Math.min(...earlier.map((item) => item.deviation))) {
    return ["best", "Лучший в серии", sub]
  }
  if (hit.score === "X") return ["", "В яблочко", sub]
  return ["", "Попадание", sub]
}

function scoreLabel(score) {
  if (score === "X") return "X"
  if (score === "0" || score === 0) return "мимо"
  return String(score)
}

function compareText(hit, earlier) {
  if (!earlier.length) return ""
  const prev = earlier[earlier.length - 1].deviation
  let tail = ""
  if (hit.deviation + 0.5 < prev) tail = " · лучше предыдущего"
  else if (hit.deviation > prev + 0.5) tail = " · хуже предыдущего"
  if (hit.deviation + 0.5 < Math.min(...earlier.map((item) => item.deviation))) tail = " · лучший в серии"
  return tail
}

function sessionLabel(item) {
  const date = new Date(item.started_at).toLocaleString("ru-RU", {
    day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  })
  const tail = item.status === "open" ? "идёт" : "закрыта"
  return `Сессия ${item.number} · ${date} · ${tail}`
}

function visibleHits() {
  if (!state) return []
  if (reviewIndex == null) return state.hits
  return state.hits.slice(0, reviewIndex)
}

function currentView() {
  if (!$("video").hidden) return $("video")
  if (!$("preview").hidden) return $("preview")
  return null
}

function mapClick(event, el) {
  if (!el) return null
  const map = viewMap(el)
  if (!map) return null
  const rect = $("view").getBoundingClientRect()
  const [x, y] = map.toNatural(event.clientX - rect.left, event.clientY - rect.top)
  if (x < 0 || y < 0 || x > map.nw || y > map.nh) return null
  return { x, y, w: map.nw, h: map.nh }
}

function activeCalib() {
  if (mode === "demo") return demoCalib
  return state ? state.calib : null
}

function zoomActive() {
  return settings.zoom && !calibStep && !!activeCalib()
}

function viewSize() {
  const rect = $("view").getBoundingClientRect()
  return { width: rect.width, height: rect.height }
}

function viewMap(el, rect = viewSize()) {
  const base = viewBox(el, rect)
  if (!base) return null
  const calib = activeCalib()
  let target = null
  if (calib) {
    const [fw, fh] = calib.frame
    const sx = base.nw / fw
    const sy = base.nh / fh
    target = {
      x: calib.center[0] * sx,
      y: calib.center[1] * sy,
      r: calib.outer_r * (sx + sy) / 2,
    }
  }
  let k = 1
  let tx = 0
  let ty = 0
  let zoom = base.scale
  let ox = base.ox
  let oy = base.oy
  if (target && zoomActive()) {
    zoom = Math.min(rect.width, rect.height) / (target.r * ZOOM_CROP)
    k = zoom / base.scale
    tx = rect.width / 2 - (base.ox + target.x * base.scale) * k
    ty = rect.height / 2 - (base.oy + target.y * base.scale) * k
    ox = rect.width / 2 - target.x * zoom
    oy = rect.height / 2 - target.y * zoom
  }
  return {
    nw: base.nw,
    nh: base.nh,
    target,
    k, tx, ty, ox, oy,
    scale: zoom,
    toScreen: (x, y) => [ox + x * zoom, oy + y * zoom],
    toNatural: (x, y) => [(x - ox) / zoom, (y - oy) / zoom],
  }
}

function applyZoom() {
  for (const el of [$("video"), $("preview")]) {
    const map = viewMap(el)
    el.style.transform = map && map.k !== 1 ? `translate(${map.tx}px, ${map.ty}px) scale(${map.k})` : ""
  }
  $("btn-zoom").classList.toggle("on", settings.zoom)
}

function send(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload))
}

function paintRecord() {
  const source = currentView()
  if (!source) return
  const canvas = $("record-canvas")
  const rect = { width: canvas.width, height: canvas.height }
  const map = viewMap(source, rect)
  if (!map) return
  const ctx = canvas.getContext("2d")
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.fillStyle = "#000"
  ctx.fillRect(0, 0, rect.width, rect.height)
  ctx.drawImage(source, map.ox, map.oy, map.nw * map.scale, map.nh * map.scale)
  drawScene(ctx, map, 1.6)
  drawRecordBanner(ctx, rect)
}

function drawRecordBanner(ctx, rect) {
  if ($("banner").hidden) return
  const title = $("banner-title").textContent
  const sub = $("banner-sub").textContent
  const kind = $("banner").className
  const cy = rect.height / 2
  const band = ctx.createLinearGradient(0, 0, rect.width, 0)
  band.addColorStop(0, "rgba(0,0,0,0)")
  band.addColorStop(0.18, "rgba(0,0,0,0.78)")
  band.addColorStop(0.82, "rgba(0,0,0,0.78)")
  band.addColorStop(1, "rgba(0,0,0,0)")
  ctx.fillStyle = band
  ctx.fillRect(0, cy - 80, rect.width, 160)
  ctx.textAlign = "center"
  ctx.lineJoin = "round"
  ctx.font = "700 92px Oswald, sans-serif"
  ctx.lineWidth = 8
  ctx.strokeStyle = "#000"
  ctx.strokeText(title.toUpperCase(), rect.width / 2, cy + 20)
  ctx.fillStyle = kind.includes("miss") ? "#e23b3b" : kind.includes("best") ? "#79d27a" : kind.includes("start") ? "#f4f1ea" : "#fcaf17"
  ctx.fillText(title.toUpperCase(), rect.width / 2, cy + 20)
  ctx.font = "500 28px Oswald, sans-serif"
  ctx.lineWidth = 5
  ctx.strokeText(sub.toUpperCase(), rect.width / 2, cy + 62)
  ctx.fillStyle = "#f4f1ea"
  ctx.fillText(sub.toUpperCase(), rect.width / 2, cy + 62)
}

function showPreview(blob) {
  const url = URL.createObjectURL(blob)
  $("preview").src = url
  $("preview").hidden = false
  $("video").hidden = true
  $("placeholder").hidden = true
  if (previewUrl) URL.revokeObjectURL(previewUrl)
  previewUrl = url
}

async function saveSettings(patchBody) {
  settings = { ...settings, ...patchBody }
  renderSettings()
  scheduleOverlay()
  if (state) drawSheet(visibleHits())
  try {
    settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(patchBody) })
  } catch (error) {
    setStatus(`Настройки не сохранились: ${error.message}`)
  }
}

function renderSettings() {
  $("set-record").checked = !!settings.record
  $("set-server").checked = !!settings.server_video
  $("set-server").disabled = !settings.record
  $("set-zoom").checked = !!settings.zoom
  $("set-label").value = String(settings.hit_label_px)
  $("set-label-val").textContent = String(settings.hit_label_px)
  $("set-label-sample").style.fontSize = `${settings.hit_label_px}px`
  $("btn-zoom").classList.toggle("on", !!settings.zoom)
}

function openSettings() {
  renderSettings()
  $("settings").hidden = false
  $("settings-close").focus()
}

function closeSettings() {
  $("settings").hidden = true
}

function recorderMime() {
  if (!window.MediaRecorder) return ""
  return ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find((type) => MediaRecorder.isTypeSupported(type)) || ""
}

function startRecording() {
  if (recorder && recorder.state === "paused") {
    recorder.resume()
    return
  }
  if (!settings.record || recorder || !window.MediaRecorder) return
  paintRecord()
  const canvas = $("record-canvas")
  const stream = canvas.captureStream(RECORD_FPS)
  chunks = []
  const mime = recorderMime()
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size) chunks.push(event.data)
  }
  recorder.start(1000)
}

function stopRecording() {
  if (!recorder || recorder.state === "inactive") return Promise.resolve()
  const current = recorder
  return new Promise((resolve) => {
    current.addEventListener("stop", () => {
      const blob = new Blob(chunks, { type: current.mimeType || "video/webm" })
      if (blob.size) parts.push(blob)
      if (recorder === current) recorder = null
      resolve()
    }, { once: true })
    current.stop()
  })
}

function partName(index, total) {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "").slice(0, 15)
  const suffix = total > 1 ? `_${index + 1}` : ""
  return `session_${state.number}_${stamp}${suffix}.webm`
}

function bindDownload(blob, name) {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl)
  downloadUrl = URL.createObjectURL(blob)
  $("download-link").href = downloadUrl
  $("download-link").download = name
  $("download-link").hidden = false
  $("download-link").textContent = `Скачать ${name}`
}

async function uploadPart(blob, name) {
  const body = new FormData()
  body.append("file", blob, name)
  setStatus("Загружаю видео на сервер…")
  const res = await fetch(`/api/sessions/${state.id}/video`, { method: "POST", body })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || "Не удалось сохранить на сервер")
  state = data
}

async function saveParts() {
  await stopRecording()
  if (!parts.length) return
  const blobs = parts.splice(0, parts.length)
  const names = blobs.map((blob, index) => partName(index, blobs.length))
  blobs.forEach((blob, index) => bindDownload(blob, names[index]))
  try { $("download-link").click() } catch { /* браузер может отложить скачивание */ }
  if (settings.server_video) {
    try {
      await uploadPart(blobs[blobs.length - 1], names[names.length - 1])
      setStatus("Видео на сервере. Ссылка на скачивание — в «Настройках»")
    } catch (error) {
      setStatus(`${error.message}. Файл можно скачать ссылкой.`)
    }
  } else {
    try {
      state = await api(`/api/sessions/${state.id}`, {
        method: "PATCH",
        body: JSON.stringify({ video_store: "local", video_name: names[0] }),
      })
    } catch { /* имя файла не критично */ }
    setStatus("Видео скачивается на этот компьютер")
  }
}

function startSending() {
  if (sendTimer) return
  const canvas = document.createElement("canvas")
  sendTimer = setInterval(async () => {
    if (!running || mode !== "browser" || !$("video").videoWidth || !ws || ws.readyState !== WebSocket.OPEN) return
    const now = performance.now()
    if (inFlight && now - inFlightAt < 3000) return
    if (now - inFlightAt < FRAME_GAP_MS) return
    inFlight = true
    inFlightAt = now
    const video = $("video")
    const map = viewMap(video)
    let header = null
    if (map && map.target && !calibStep) {
      const side = map.target.r * ZOOM_CROP
      const out = Math.max(64, Math.round(Math.min(side, CROP_OUT)))
      const x0 = map.target.x - side / 2
      const y0 = map.target.y - side / 2
      canvas.width = out
      canvas.height = out
      const ctx = canvas.getContext("2d")
      ctx.fillStyle = "#000"
      ctx.fillRect(0, 0, out, out)
      ctx.drawImage(video, x0, y0, side, side, 0, 0, out, out)
      header = cropHeader(x0, y0, side, map.nw, map.nh)
    } else {
      const width = Math.min(960, video.videoWidth)
      canvas.width = width
      canvas.height = Math.max(1, Math.round(video.videoHeight * width / video.videoWidth))
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)
    }
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9))
    if (blob && ws && ws.readyState === WebSocket.OPEN && running && mode === "browser") {
      const jpeg = await blob.arrayBuffer()
      ws.send(header ? await new Blob([header, jpeg]).arrayBuffer() : jpeg)
    } else {
      inFlight = false
    }
  }, 30)
}

function cropHeader(x0, y0, side, fullW, fullH) {
  const buf = new ArrayBuffer(24)
  const bytes = new Uint8Array(buf)
  bytes.set([0x53, 0x54, 0x43, 0x31])
  const view = new DataView(buf)
  ;[x0, y0, side, fullW, fullH].forEach((value, index) => view.setFloat32(4 + index * 4, value, true))
  return buf
}

function onTick(msg) {
  inFlight = false
  const now = performance.now()
  tickTimes.push(now)
  while (tickTimes.length && now - tickTimes[0] > 2000) tickTimes.shift()
  if (!running || reviewIndex != null || calibStep || now < quietUntil) return
  const fps = tickTimes.length / 2
  const text = LIVE[msg.status] || STATUS[msg.status] || msg.status
  setStatus(`${text} · ${fps.toFixed(0)} к/с · ${msg.ms} мс`)
}

function takeSnap() {
  const source = currentView()
  if (!source) return null
  const width = source.videoWidth || source.naturalWidth
  const height = source.videoHeight || source.naturalHeight
  if (!width || !height) return null
  const canvas = document.createElement("canvas")
  canvas.width = Math.min(640, width)
  canvas.height = Math.max(1, Math.round(height * canvas.width / width))
  canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL("image/jpeg", 0.8)
}

function showSnap(hit) {
  const snap = hit && snaps[hit.n]
  if (!snap) return
  $("preview").src = snap
  $("preview").hidden = false
  $("video").hidden = true
  $("placeholder").hidden = true
}

async function refreshHistory() {
  const list = await api("/api/sessions")
  $("history").replaceChildren()
  for (const item of list) {
    const option = document.createElement("option")
    option.value = String(item.id)
    option.textContent = sessionLabel(item)
    $("history").appendChild(option)
  }
  if (state) $("history").value = String(state.id)
}

function render() {
  if (!state) return
  const stats = state.stats
  $("stat-count").textContent = String(stats.count)
  $("stat-sum").textContent = String(stats.total)
  $("stat-avg").textContent = stats.average == null ? "—" : String(stats.average)
  $("stat-dev").textContent = stats.mean_deviation == null ? "—" : `${Math.round(stats.mean_deviation)}%`
  $("session-no").textContent = String(state.number)
  $("sens-val").textContent = String(state.sensitivity)
  $("size-val").textContent = Number(state.size_scale).toFixed(1)
  if (document.activeElement !== $("note")) $("note").value = state.note || ""
  $("chk-carry").checked = !!state.carry_note
  if (document.activeElement !== $("target")) $("target").value = state.target_key
  const closed = state.status !== "open"
  $("btn-start").disabled = closed
  $("btn-start").textContent = running ? "Стоп" : "Старт"
  $("btn-start").classList.toggle("on", running)
  const shown = visibleHits()
  if (reviewIndex == null) $("step-label").textContent = ""
  else if (reviewIndex <= 0) $("step-label").textContent = "до первого"
  else {
    const hit = state.hits[reviewIndex - 1]
    const earlier = state.hits.slice(0, reviewIndex - 1)
    $("step-label").textContent = `${reviewIndex}/${state.hits.length} · ${Math.round(hit.deviation)}%${compareText(hit, earlier)}`
  }
  renderGroup(shown)
  renderLog()
  renderBars(shown)
  drawSheet(shown)
  scheduleOverlay()
}

function cancelCalib() {
  calibStep = null
  pendingCenter = null
  mousePoint = null
  $("view").classList.remove("calibrating")
  setStatus("Калибровка отменена")
  scheduleOverlay()
}

function scheduleOverlay() {
  if (overlayQueued) return
  overlayQueued = true
  requestAnimationFrame(() => {
    overlayQueued = false
    drawOverlay()
  })
}

function viewBox(el, rect = viewSize()) {
  const nw = el.videoWidth || el.naturalWidth
  const nh = el.videoHeight || el.naturalHeight
  if (!nw || !nh) return null
  const scale = Math.min(rect.width / nw, rect.height / nh)
  return {
    nw, nh, scale,
    ox: (rect.width - nw * scale) / 2,
    oy: (rect.height - nh * scale) / 2,
  }
}

function drawOverlay() {
  const canvas = $("overlay")
  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.max(1, Math.round(rect.width * dpr))
  canvas.height = Math.max(1, Math.round(rect.height * dpr))
  const ctx = canvas.getContext("2d")
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, rect.width, rect.height)
  applyZoom()
  const el = currentView()
  if (!el) return
  const box = viewMap(el)
  if (!box) return
  const toScreen = box.toScreen

  if (calibStep) {
    if (calibStep === "edge" && pendingCenter) {
      const [cx, cy] = toScreen(pendingCenter.x * box.nw / pendingCenter.w, pendingCenter.y * box.nh / pendingCenter.h)
      cross(ctx, cx, cy, 14, "#fcaf17")
      if (mousePoint) {
        const [mx, my] = toScreen(mousePoint.x, mousePoint.y)
        const radius = Math.hypot(mx - cx, my - cy)
        ring(ctx, cx, cy, radius, "#fcaf17", 2.5)
        ctx.setLineDash([6, 6])
        ctx.beginPath()
        ctx.moveTo(cx, cy)
        ctx.lineTo(mx, my)
        ctx.strokeStyle = "rgba(252, 175, 23, 0.8)"
        ctx.lineWidth = 1.5
        ctx.stroke()
        ctx.setLineDash([])
      }
    } else if (mousePoint) {
      const [mx, my] = toScreen(mousePoint.x, mousePoint.y)
      cross(ctx, mx, my, 18, "#fcaf17")
    }
    return
  }

  drawScene(ctx, box, 1)
}

function drawScene(ctx, map, scale) {
  if (!map.target || !state) return
  const [cx, cy] = map.toScreen(map.target.x, map.target.y)
  const outer = map.target.r * map.scale
  const target = targets.find((item) => item.key === state.target_key)
  if (target) {
    for (const item of target.rings) {
      ring(ctx, cx, cy, item.radius * outer, "rgba(252, 175, 23, 0.45)", 1 * scale)
    }
  }
  ring(ctx, cx, cy, outer, "#fcaf17", 2.5 * scale)
  cross(ctx, cx, cy, 10 * scale, "#fcaf17")
  const label = settings.hit_label_px * scale
  const dot = Math.max(5, label * 0.42)
  const hits = visibleHits()
  hits.forEach((hit, index) => {
    const x = cx + hit.norm_x * outer
    const y = cy + hit.norm_y * outer
    const hot = reviewIndex != null && index === hits.length - 1
    ctx.beginPath()
    ctx.arc(x, y, hot ? dot * 1.3 : dot, 0, Math.PI * 2)
    ctx.fillStyle = hot ? "rgba(252, 175, 23, 0.95)" : "rgba(226, 59, 59, 0.85)"
    ctx.fill()
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1.5 * scale
    ctx.stroke()
    ctx.font = `600 ${label}px Oswald, sans-serif`
    ctx.textAlign = "left"
    ctx.lineJoin = "round"
    ctx.lineWidth = Math.max(3, label * 0.22)
    ctx.strokeStyle = "#000"
    ctx.strokeText(String(index + 1), x + dot + 2, y - dot - 2)
    ctx.fillStyle = "#fcaf17"
    ctx.fillText(String(index + 1), x + dot + 2, y - dot - 2)
  })
}

function ring(ctx, x, y, radius, color, width) {
  if (radius < 1) return
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.strokeStyle = "rgba(0, 0, 0, 0.6)"
  ctx.lineWidth = width + 2
  ctx.stroke()
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.stroke()
}

function cross(ctx, x, y, size, color) {
  ctx.beginPath()
  ctx.moveTo(x - size, y)
  ctx.lineTo(x + size, y)
  ctx.moveTo(x, y - size)
  ctx.lineTo(x, y + size)
  ctx.strokeStyle = "#000"
  ctx.lineWidth = 4
  ctx.stroke()
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.stroke()
}

function renderBars(shown) {
  const target = targets.find((item) => item.key === state.target_key)
  const numeric = target ? target.rings.map((ring) => ring.score).filter((score) => typeof score === "number") : []
  const top = numeric.length ? Math.max(...numeric) : 10
  let score = 0
  let tight = 0
  if (shown.length) {
    const points = shown.map((hit) => (hit.score === "X" ? top : Number(hit.score)))
    score = points.reduce((sum, value) => sum + value, 0) / shown.length / top * 100
    const deviation = shown.reduce((sum, hit) => sum + hit.deviation, 0) / shown.length
    tight = Math.max(0, 100 - deviation)
  }
  $("bar-score").style.width = `${Math.min(100, score)}%`
  $("bar-tight").style.width = `${Math.min(100, tight)}%`
}

function renderGroup(shown) {
  if (shown.length < 2 || !state.group || shown.length === state.hits.length) {
    if (!state.group || shown.length < 2) {
      $("group").textContent = "Нужно минимум два попадания"
      $("group").className = ""
      return
    }
  }
  const group = state.group
  $("group").textContent = `Разброс ${group.spread.toFixed(3)}\nСТП ${group.avg_x.toFixed(2)} / ${group.avg_y.toFixed(2)}\n${group.direction}`
  $("group").className = group.centered ? "good" : "bad"
}

function renderLog() {
  const list = $("hit-log")
  list.replaceChildren()
  const current = reviewIndex == null ? null : reviewIndex - 1
  for (const hit of [...state.hits].reverse()) {
    const item = document.createElement("li")
    if (current === hit.n - 1) item.className = "on"
    const n = document.createElement("span")
    n.className = "muted"
    n.textContent = String(hit.n).padStart(2, "0")
    const score = document.createElement("span")
    score.textContent = scoreLabel(hit.score)
    const pct = document.createElement("span")
    pct.textContent = `${Math.round(hit.deviation)}%`
    pct.className = hit.deviation < 15 ? "good" : hit.deviation > 50 ? "bad" : ""
    const time = document.createElement("span")
    time.className = "muted"
    time.textContent = new Date(hit.at).toLocaleTimeString("ru-RU")
    item.append(n, score, pct, time)
    item.addEventListener("click", () => jump(hit.n))
    list.appendChild(item)
  }
  if (!state.hits.length) {
    const item = document.createElement("li")
    item.textContent = "Пока пусто — клик по схеме тоже считается"
    list.appendChild(item)
  }
}

function drawSheet(shown) {
  const canvas = $("sheet")
  const rect = canvas.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.max(1, Math.round(rect.width * dpr))
  canvas.height = Math.max(1, Math.round(rect.height * dpr))
  const ctx = canvas.getContext("2d")
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const target = targets.find((item) => item.key === state.target_key) || targets[0]
  if (!target) return
  const side = Math.min(canvas.width, canvas.height)
  const radius = side * 0.44
  const cx = canvas.width / 2
  const cy = canvas.height / 2
  sheetGeom = { cx, cy, radius }
  const rings = [...target.rings].reverse()
  for (const ring of rings) {
    ctx.beginPath()
    ctx.arc(cx, cy, Math.max(1, ring.radius * radius), 0, Math.PI * 2)
    ctx.fillStyle = ring.color
    ctx.fill()
  }
  for (const ring of target.rings) {
    const rr = ring.radius * radius
    ctx.beginPath()
    ctx.arc(cx, cy, rr, 0, Math.PI * 2)
    ctx.strokeStyle = ring.color === "#161616" || ring.color === "#202020" ? "#ececec" : "#1a1a1a"
    ctx.lineWidth = Math.max(1, dpr)
    ctx.stroke()
    if (ring.score !== "X" && rr > 22 * dpr) {
      ctx.fillStyle = ring.color === "#ececec" ? "#141414" : "#f2f2f2"
      ctx.font = `${Math.round(12 * dpr)}px sans-serif`
      ctx.textAlign = "center"
      ctx.fillText(String(ring.score), cx, cy - rr + 14 * dpr)
    }
  }
  const xRing = target.rings[0]
  if (xRing && xRing.score === "X") {
    ctx.fillStyle = xRing.color === "#ececec" ? "#141414" : "#f4f4f4"
    ctx.font = `${Math.round(14 * dpr)}px sans-serif`
    ctx.textAlign = "center"
    ctx.fillText("X", cx, cy + 5 * dpr)
  }
  shown.forEach((hit, index) => {
    const x = cx + hit.norm_x * radius
    const y = cy + hit.norm_y * radius
    const hot = reviewIndex != null ? index === shown.length - 1 : false
    ctx.beginPath()
    ctx.arc(x, y, (hot ? 8 : 6) * dpr, 0, Math.PI * 2)
    ctx.fillStyle = hot ? "#fcaf17" : "#e23b3b"
    ctx.fill()
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2 * dpr
    ctx.stroke()
    const label = Math.round(settings.hit_label_px * 0.85 * dpr)
    ctx.font = `600 ${label}px Oswald, sans-serif`
    ctx.textAlign = "left"
    ctx.lineJoin = "round"
    ctx.lineWidth = Math.max(3, label * 0.22)
    ctx.strokeStyle = "#000"
    ctx.strokeText(String(index + 1), x + 9 * dpr, y - 9 * dpr)
    ctx.fillStyle = "#fcaf17"
    ctx.fillText(String(index + 1), x + 9 * dpr, y - 9 * dpr)
  })
  if (shown.length >= 2) {
    const mx = shown.reduce((sum, hit) => sum + hit.norm_x, 0) / shown.length
    const my = shown.reduce((sum, hit) => sum + hit.norm_y, 0) / shown.length
    ctx.strokeStyle = "#79d27a"
    ctx.lineWidth = 3 * dpr
    ctx.beginPath()
    ctx.moveTo(cx + mx * radius - 8 * dpr, cy + my * radius)
    ctx.lineTo(cx + mx * radius + 8 * dpr, cy + my * radius)
    ctx.moveTo(cx + mx * radius, cy + my * radius - 8 * dpr)
    ctx.lineTo(cx + mx * radius, cy + my * radius + 8 * dpr)
    ctx.stroke()
  }
}

function jump(n) {
  reviewIndex = n
  if (running) send({ type: "stop" })
  running = false
  const hit = state.hits[n - 1]
  showSnap(hit)
  render()
}

function step(delta) {
  if (!state || !state.hits.length) return
  const next = Math.max(0, Math.min(state.hits.length, (reviewIndex == null ? state.hits.length : reviewIndex) + delta))
  if (next === state.hits.length) {
    reviewIndex = null
    if (mode === "browser" && cameraStream) {
      $("video").hidden = false
      $("preview").hidden = true
    }
  } else {
    reviewIndex = next
    if (running) send({ type: "stop" })
    running = false
    if (next > 0) showSnap(state.hits[next - 1])
  }
  render()
}

function applyState(next) {
  state = next
  render()
}

function onMessage(event) {
  if (event.data instanceof Blob) {
    if ((mode === "demo" || mode === "ip") && reviewIndex == null) showPreview(event.data)
    return
  }
  const msg = JSON.parse(event.data)
  if (msg.type === "tick") {
    onTick(msg)
  } else if (msg.type === "state" && msg.state) {
    applyState(msg.state)
    if (msg.mode) mode = msg.mode
  } else if (msg.type === "mode") {
    mode = msg.mode
    demoCalib = msg.calib || null
    scheduleOverlay()
    if (mode === "demo" || mode === "ip") {
      $("placeholder").hidden = true
    }
    if (mode === "browser") {
      $("preview").hidden = true
      if (cameraStream) $("video").hidden = false
    }
    setStatus(mode === "demo" ? "Демо-мишень. Нажмите «Старт»" : mode === "ip" ? "Поток камеры. Отметьте центр, если он не найден." : "Вебка в браузере")
  } else if (msg.type === "started") {
    const resumed = running
    running = true
    reviewIndex = null
    try { startRecording() } catch { /* анализ идёт и без записи */ }
    if (!resumed) {
      setStatus("Старт")
      const target = targets.find((item) => item.key === state.target_key)
      showBanner("start", `Сессия ${state.number}`, target ? target.name : "", 1600)
    }
    render()
  } else if (msg.type === "stopped") {
    running = false
    if (recorder && recorder.state === "recording") recorder.pause()
    render()
  } else if (msg.type === "status") {
    if (reviewIndex == null && performance.now() >= quietUntil) setStatus(STATUS[msg.status] || msg.status)
  } else if (msg.type === "hits") {
    const earlier = state ? state.hits.slice() : []
    for (const hit of msg.hits || []) snaps[hit.n] = takeSnap()
    if (msg.state) state = msg.state
    if (msg.hits && msg.hits.length && reviewIndex == null) {
      const hit = msg.hits[msg.hits.length - 1]
      const before = earlier.filter((item) => item.n < hit.n)
      setStatus(`Выстрел ${hit.n} · ${scoreLabel(hit.score)} · отклонение ${Math.round(hit.deviation)}%${compareText(hit, before)}`)
      showBanner(...bannerForHit(hit, before))
      quietUntil = performance.now() + 3500
    }
    render()
    refreshHistory().catch(() => {})
  } else if (msg.type === "error") {
    setStatus(msg.message)
    running = false
    render()
  }
}

function connect() {
  if (!state) return
  clearTimeout(reconnectTimer)
  if (ws) {
    ws.onclose = null
    ws.close()
  }
  inFlight = false
  const proto = location.protocol === "https:" ? "wss" : "ws"
  const socket = new WebSocket(`${proto}://${location.host}/api/live/${state.id}`)
  ws = socket
  socket.binaryType = "blob"
  socket.onmessage = onMessage
  socket.onopen = () => {
    if (mode === "browser" && cameraStream) send({ type: "browser" })
    else if (mode === "demo") send({ type: "demo" })
    else if (mode === "ip" && $("ip-url").value.trim()) send({ type: "ip", url: $("ip-url").value.trim() })
    if (running) send({ type: "start" })
  }
  socket.onclose = (event) => {
    if (ws !== socket) return
    if (event.code === 4401) {
      location.href = "/login"
      return
    }
    inFlight = false
    setStatus("Связь с сервером потеряна, переподключаюсь…")
    reconnectTimer = setTimeout(connect, 1500)
  }
}

async function patch(body) {
  state = await api(`/api/sessions/${state.id}`, { method: "PATCH", body: JSON.stringify(body) })
  render()
}

async function openCamera() {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    })
  } catch (error) {
    if (error && (error.name === "OverconstrainedError" || error.name === "NotReadableError")) {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    }
    throw error
  }
}

function cameraError(error) {
  const name = error && error.name
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Доступ к камере запрещён. Разрешите его в адресной строке браузера."
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "Камеру держит другая программа: десктопный ShotTracker, Zoom, другая вкладка. Закройте её и нажмите «Вебка» ещё раз."
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "Браузер не видит камеру. Проверьте, что она подключена."
  }
  return `Камера не открылась: ${name || "ошибка"}`
}

async function useCamera() {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop())
  cameraStream = null
  if (!window.isSecureContext) {
    setStatus("Камера в браузере работает только по https:// или на localhost")
    return
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Этот браузер не отдаёт камеру")
    return
  }
  try {
    cameraStream = await openCamera()
  } catch (error) {
    setStatus(cameraError(error))
    return
  }
  $("video").srcObject = cameraStream
  await $("video").play()
  $("video").hidden = false
  $("preview").hidden = true
  $("placeholder").hidden = true
  mode = "browser"
  send({ type: "browser" })
  setStatus("Вебка в браузере. «Центр», затем «Старт».")
}

async function frameBlob() {
  const source = currentView() || $("video")
  const width = source.videoWidth || source.naturalWidth
  const height = source.videoHeight || source.naturalHeight
  if (!width || !height) return null
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  canvas.getContext("2d").drawImage(source, 0, 0)
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85))
}

function bind() {
  $("btn-logout").onclick = async () => {
    await fetch("/api/logout", { method: "POST" })
    location.href = "/login"
  }
  $("history").onchange = async () => {
    const id = Number($("history").value)
    if (!state || id === state.id) return
    state = await api(`/api/sessions/${id}`)
    reviewIndex = null
    snaps = {}
    running = false
    connect()
    render()
  }
  $("target").onchange = () => patch({ target_key: $("target").value }).catch((error) => setStatus(error.message))
  $("note").addEventListener("change", () => patch({ note: $("note").value }).catch((error) => setStatus(error.message)))
  $("chk-carry").onchange = () => patch({ carry_note: $("chk-carry").checked }).catch((error) => setStatus(error.message))
  $("btn-copy-note").onclick = async () => {
    try {
      state = await api(`/api/sessions/${state.id}/note/copy`, { method: "POST" })
      render()
      setStatus("Заметка скопирована с прошлой сессии")
    } catch (error) {
      setStatus(error.message)
    }
  }
  $("btn-zoom").onclick = () => {
    if (!settings.zoom && !activeCalib()) setStatus("Зум включится после калибровки")
    saveSettings({ zoom: !settings.zoom })
  }
  $("btn-settings").onclick = openSettings
  $("settings-close").onclick = closeSettings
  $("settings").addEventListener("click", (event) => {
    if (event.target === $("settings")) closeSettings()
  })
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("settings").hidden) closeSettings()
  })
  $("set-record").onchange = () => {
    const on = $("set-record").checked
    saveSettings({ record: on })
    if (!on) stopRecording()
    else if (running) startRecording()
  }
  $("set-server").onchange = () => saveSettings({ server_video: $("set-server").checked })
  $("set-zoom").onchange = () => saveSettings({ zoom: $("set-zoom").checked })
  $("set-label").oninput = () => {
    const px = Number($("set-label").value)
    settings.hit_label_px = px
    renderSettings()
    scheduleOverlay()
    if (state) drawSheet(visibleHits())
  }
  $("set-label").onchange = () => saveSettings({ hit_label_px: Number($("set-label").value) })
  $("btn-center").onclick = () => {
    if (mode === "demo") {
      setStatus("В демо мишень уже размечена")
      return
    }
    if (!currentView()) {
      setStatus("Сначала включите вебку или IP-камеру")
      return
    }
    calibStep = "center"
    pendingCenter = null
    $("view").classList.add("calibrating")
    setStatus("Кликните центр мишени. Правый клик или Esc — отмена")
    scheduleOverlay()
  }
  $("view").addEventListener("mousemove", (event) => {
    if (!calibStep) return
    mousePoint = mapClick(event, currentView())
    scheduleOverlay()
  })
  $("view").addEventListener("mouseleave", () => {
    mousePoint = null
    scheduleOverlay()
  })
  $("view").addEventListener("contextmenu", (event) => {
    if (!calibStep) return
    event.preventDefault()
    cancelCalib()
  })
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && calibStep) cancelCalib()
  })
  $("view").addEventListener("click", (event) => {
    if (!calibStep) return
    const point = mapClick(event, currentView())
    if (!point) return
    if (calibStep === "center") {
      pendingCenter = point
      calibStep = "edge"
      setStatus("Кликните внешний край зоны 1. Правый клик — отмена")
      scheduleOverlay()
      return
    }
    if (Math.hypot(point.x - pendingCenter.x, point.y - pendingCenter.y) < 10) {
      setStatus("Край слишком близко к центру. Кликните по внешнему краю зоны 1")
      return
    }
    send({
      type: "calib",
      cx: pendingCenter.x, cy: pendingCenter.y,
      ex: point.x, ey: point.y,
      w: point.w, h: point.h,
    })
    calibStep = null
    pendingCenter = null
    $("view").classList.remove("calibrating")
    quietUntil = performance.now() + 2500
    setStatus("Калибровка сохранена. Проверьте кольца на кадре")
    scheduleOverlay()
  })
  $("sheet").addEventListener("click", (event) => {
    if (!state || state.status !== "open" || reviewIndex != null || !sheetGeom) return
    const rect = $("sheet").getBoundingClientRect()
    const x = (event.clientX - rect.left) * $("sheet").width / rect.width
    const y = (event.clientY - rect.top) * $("sheet").height / rect.height
    const normX = (x - sheetGeom.cx) / sheetGeom.radius
    const normY = (y - sheetGeom.cy) / sheetGeom.radius
    if (Math.hypot(normX, normY) > 1.15) return
    send({ type: "manual", norm_x: normX, norm_y: normY })
  })
  $("btn-auto").onclick = async () => {
    const blob = await frameBlob()
    if (!blob) {
      setStatus("Нет кадра для автопоиска")
      return
    }
    const body = new FormData()
    body.append("file", blob, "frame.jpg")
    try {
      state = await api(`/api/sessions/${state.id}/auto-calib`, { method: "POST", body })
      render()
      setStatus("Круг найден")
    } catch (error) {
      setStatus(error.message)
    }
  }
  $("btn-reference").onclick = () => send({ type: "reference" })
  $("btn-start").onclick = () => {
    if (state.status !== "open") return
    if (running) {
      send({ type: "stop" })
      return
    }
    reviewIndex = null
    send({ type: "start" })
  }
  $("btn-new").onclick = async () => {
    try {
      await saveParts()
      const finished = state.stats
      if (finished.count) {
        showBanner("best", "Сессия завершена", `сумма ${finished.total} · отклонение ${Math.round(finished.mean_deviation)}%`, 2600)
      }
      state = await api("/api/sessions", { method: "POST" })
      reviewIndex = null
      snaps = {}
      running = false
      await refreshHistory()
      connect()
      render()
      setStatus("Новая сессия")
    } catch (error) {
      setStatus(error.message)
    }
  }
  $("btn-undo").onclick = async () => {
    try {
      state = await api(`/api/sessions/${state.id}/undo`, { method: "POST" })
      render()
    } catch (error) {
      setStatus(error.message)
    }
  }
  $("btn-demo").onclick = () => {
    mode = "demo"
    send({ type: "demo" })
  }
  $("btn-camera").onclick = () => useCamera()
  $("btn-ip").onclick = () => {
    const url = $("ip-url").value.trim()
    if (!url) {
      setStatus("Вставьте адрес камеры")
      return
    }
    mode = "ip"
    send({ type: "ip", url })
  }
  $("btn-step-prev").onclick = () => step(-1)
  $("btn-step-next").onclick = () => step(1)
  $("sens-dec").onclick = () => patch({ sensitivity: state.sensitivity - 2 }).catch((error) => setStatus(error.message))
  $("sens-inc").onclick = () => patch({ sensitivity: state.sensitivity + 2 }).catch((error) => setStatus(error.message))
  $("size-dec").onclick = () => patch({ size_scale: Math.round((state.size_scale - 0.1) * 10) / 10 }).catch((error) => setStatus(error.message))
  $("size-inc").onclick = () => patch({ size_scale: Math.round((state.size_scale + 0.1) * 10) / 10 }).catch((error) => setStatus(error.message))
  $("btn-save-video").onclick = () => saveParts().catch((error) => setStatus(error.message))
  window.addEventListener("resize", () => {
    if (state) drawSheet(visibleHits())
    scheduleOverlay()
  })
  $("video").addEventListener("loadedmetadata", scheduleOverlay)
  $("preview").addEventListener("load", scheduleOverlay)
  setInterval(() => {
    if (recorder && recorder.state === "recording") paintRecord()
  }, 1000 / RECORD_FPS)
  startSending()
}

async function boot() {
  const me = await api("/api/me")
  $("user-name").textContent = me.username
  settings = { ...settings, ...(me.settings || {}) }
  targets = await api("/api/targets")
  for (const target of targets) {
    const option = document.createElement("option")
    option.value = target.key
    option.textContent = target.name
    $("target").appendChild(option)
  }
  state = await api("/api/sessions/current")
  bind()
  renderSettings()
  await refreshHistory()
  render()
  connect()
  setStatus("Выберите вебку, демо или IP-камеру")
}

boot().catch((error) => setStatus(error.message || "Не удалось открыть сессию"))
