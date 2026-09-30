// Only presentation state changes here. Native Gradio tabs own page selection,
// event loading, inference callbacks, media components and all session state.
const root = element.closest('#muan-interface');
if (root && !root.dataset.muanNavigationReady) {
  root.dataset.muanNavigationReady = 'true';
  root.dataset.muanView = 'welcome';
  root.dataset.muanMenu = 'closed';
  let workspaceScroll = 0;
  let resizeFrame = 0;
  const resizingPlots = new WeakSet();
  // Plotly initially measures 700px when its Gradio parent is hidden. Resize
  // only visible, initialized plots whose measured width actually changed.
  const resizeCharts = () => {
    if (resizeFrame) return;
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = 0;
      if (root.dataset.muanView !== 'workspace' || !window.Plotly?.Plots?.resize) return;
      root.querySelectorAll('.js-plotly-plot').forEach(plot => {
        if (!plot._fullLayout || !plot.getClientRects().length || plot.clientWidth < 1 ||
            Math.abs(plot.clientWidth - plot._fullLayout.width) < 2 || resizingPlots.has(plot)) return;
        resizingPlots.add(plot);
        window.Plotly.Plots.resize(plot).catch(() => {
          // A user may hide this tab again before Plotly's deferred resize.
        }).finally(() => resizingPlots.delete(plot));
      });
    });
  };
  new ResizeObserver(resizeCharts).observe(root);
  new MutationObserver(resizeCharts).observe(root, {childList: true, subtree: true});

  const get = id => root.querySelector('#' + id);
  const toolViews = {
    result: ['解读结果', 'ai-result-start'],
    record: ['总结记录', 'ai-record-start'],
    knowledge: ['知识与引用', 'ai-tools-toggle']
  };
  const setAssistantTools = (open, view = root.dataset.aiToolsView || 'knowledge') => {
    root.dataset.aiToolsView = view;
    root.dataset.aiToolsOpen = String(open);
    const title = get('ai-tools-title');
    if (title) title.textContent = toolViews[view][0];
    Object.entries(toolViews).forEach(([key, [, id]]) => {
      get(id)?.setAttribute('aria-expanded', String(open && key === view));
      get(id)?.setAttribute('aria-controls', 'ai-tools-drawer');
    });
    const drawer = get('ai-tools-drawer');
    if (drawer) drawer.scrollTop = 0;
    requestAnimationFrame(() => (open ? get('ai-tools-close') : get(toolViews[view][1]))?.focus({preventScroll:true}));
  };
  // Read native display state; do not change values or detection callbacks.
  let presentationFrame = 0;
  const syncPresentation = () => {
    if (presentationFrame) return;
    presentationFrame = requestAnimationFrame(() => {
      presentationFrame = 0;
      const metrics = get('video-metrics');
      const waiting = get('upload-status')?.textContent.includes('请选择视频后开始检测。');
      if (metrics) metrics.dataset.waiting = String(!!waiting);
      const video = root.querySelector('#result-video video');
      const stage = get('video-stage');
      if (stage) stage.dataset.hasResult = String(!!video?.getAttribute('src'));
      const composer = get('ai-composer');
      if (composer?.getClientRects().length) {
        get('muan-ai-panel')?.style.setProperty('--ai-composer-height',
          (composer.getBoundingClientRect().height + 40) + 'px');
      }
    });
  };
  new MutationObserver(syncPresentation).observe(root, {childList:true, subtree:true, characterData:true});
  new ResizeObserver(syncPresentation).observe(root);
  requestAnimationFrame(() => {
    if (get('ai-composer')) new ResizeObserver(syncPresentation).observe(get('ai-composer'));
    syncPresentation();
  });
  // Window geometry is local presentation state, independent of chat/session data.
  let assistantBounds = null;
  let assistantGesture = null;
  const compactAssistant = () => window.matchMedia('(max-width: 760px)').matches;
  const clamp = (value, min, max) => Math.max(min, Math.min(value, max));
  const fitAssistant = bounds => {
    const maxWidth = Math.max(1, window.innerWidth - 24);
    const maxHeight = Math.max(1, window.innerHeight - 24);
    const width = clamp(bounds.width, Math.min(420, maxWidth), maxWidth);
    const height = clamp(bounds.height, Math.min(520, maxHeight), maxHeight);
    return {width, height, x: clamp(bounds.x, 12, Math.max(12, window.innerWidth - width - 12)),
      y: clamp(bounds.y, 12, Math.max(12, window.innerHeight - height - 12))};
  };
  const defaultAssistant = () => fitAssistant({
    x: (parseFloat(getComputedStyle(root).getPropertyValue('--muan-sidebar-width')) || 224) + 16,
    y: 32, width: 720, height: Math.min(860, window.innerHeight - 64)
  });
  const placeAssistant = bounds => {
    const panel = get('muan-ai-panel');
    if (!panel) return;
    if (compactAssistant()) {
      panel.style.setProperty('--ai-chat-height', '260px');
      return;
    }
    assistantBounds = fitAssistant(bounds || assistantBounds || defaultAssistant());
    const {x, y, width, height} = assistantBounds;
    Object.entries({'--ai-window-x': x, '--ai-window-y': y, '--ai-window-width': width,
      '--ai-window-height': height, '--ai-chat-height': clamp(height * .43, 240, 520)})
      .forEach(([name, value]) => panel.style.setProperty(name, value + 'px'));
  };
  const endAssistantGesture = () => {
    if (!assistantGesture) return;
    const {handle, pointerId} = assistantGesture;
    assistantGesture = null;
    delete root.dataset.aiWindowAction;
    if (handle.hasPointerCapture(pointerId)) handle.releasePointerCapture(pointerId);
  };
  root.addEventListener('pointerdown', event => {
    if (!(event.target instanceof Element) || event.button !== 0 || assistantGesture || compactAssistant()) return;
    const handle = event.target.closest('#muan-ai-drag, #muan-ai-resize');
    if (!handle || root.dataset.aiOpen !== 'true') return;
    placeAssistant();
    const action = handle.id === 'muan-ai-drag' ? 'move' : 'resize';
    assistantGesture = {handle, action, pointerId: event.pointerId, startX: event.clientX,
      startY: event.clientY, bounds: {...assistantBounds}};
    root.dataset.aiWindowAction = action;
    handle.setPointerCapture(event.pointerId);
    handle.focus({preventScroll: true});
    event.preventDefault();
  });
  root.addEventListener('pointermove', event => {
    const gesture = assistantGesture;
    if (!gesture || event.pointerId !== gesture.pointerId) return;
    const dx = event.clientX - gesture.startX, dy = event.clientY - gesture.startY;
    const b = gesture.bounds;
    placeAssistant(gesture.action === 'move' ? {...b, x: b.x + dx, y: b.y + dy} : {...b,
      width: Math.min(b.width + dx, window.innerWidth - b.x - 12),
      height: Math.min(b.height + dy, window.innerHeight - b.y - 12)});
    event.preventDefault();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    root.addEventListener(name, event => {
      if (assistantGesture?.pointerId === event.pointerId) endAssistantGesture();
    });
  }
  window.addEventListener('blur', endAssistantGesture);
  window.addEventListener('resize', () => {
    endAssistantGesture();
    placeAssistant();
  });
  const setAssistant = open => {
    endAssistantGesture();
    root.dataset.aiOpen = String(open);
    get('muan-ai-launch')?.setAttribute('aria-expanded', String(open));
    get('muan-ai-launch')?.setAttribute('aria-label', open ? '关闭暮安小助手' : '打开暮安小助手');
    const panel = get('muan-ai-panel');
    panel?.setAttribute('role', 'region');
    panel?.setAttribute('aria-label', '暮安小助手');
    if (!open) {
      root.dataset.aiToolsOpen = 'false';
      Object.values(toolViews).forEach(([, id]) => get(id)?.setAttribute('aria-expanded', 'false'));
    }
    requestAnimationFrame(() => {
      if (open) placeAssistant();
      syncPresentation();
      (open ? root.querySelector('#ai-question textarea') : get('muan-ai-launch'))?.focus({preventScroll: true});
    });
  };
  const tabButtons = () => Array.from(root.querySelectorAll('#workspace-tabs [role="tab"]'));
  const selectedTab = () => tabButtons().find(button => button.getAttribute('aria-selected') === 'true' || button.classList.contains('selected'));
  const narrow = () => window.matchMedia('(max-width: 760px)').matches;
  const setMenu = open => {
    root.dataset.muanMenu = open ? 'open' : 'closed';
    get('muan-menu')?.setAttribute('aria-expanded', String(open));
  };
  const syncTab = () => {
    const tab = selectedTab();
    if (tab && get('muan-current-page')) get('muan-current-page').textContent = tab.textContent.trim();
    const tablist = root.querySelector('#workspace-tabs [role="tablist"]');
    tablist?.setAttribute('aria-orientation', 'vertical');
    tablist?.setAttribute('aria-label', '智护工作台功能导航');
    if (tablist?.id) get('muan-menu')?.setAttribute('aria-controls', tablist.id);
  };
  const cameraRunning = () => !!root.querySelector('#status-panel')?.textContent.includes('Astra监测中') || Array.from(root.querySelectorAll('#camera-input video')).some(video =>
    video.srcObject && typeof video.srcObject.getTracks === 'function' &&
    video.srcObject.getTracks().some(track => track.readyState === 'live')
  );
  const announce = message => {
    const status = get('muan-navigation-message');
    if (status) { status.textContent = message; status.hidden = !message; }
  };

  root.addEventListener('click', event => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest('button');
    if (!button) return;
    if (button.id === 'ai-tools-toggle') {
      setAssistantTools(root.dataset.aiToolsOpen !== 'true' || root.dataset.aiToolsView !== 'knowledge', 'knowledge');
    } else if (button.id === 'ai-tools-close') {
      setAssistantTools(false);
    } else if (button.id === 'ai-result-start' || button.id === 'ai-record-start') {
      setAssistantTools(true, button.id === 'ai-result-start' ? 'result' : 'record');
    } else if (button.id === 'muan-kb-open') {
      root.dataset.kbOpen = 'true';
      get('kb-refresh')?.click();
      requestAnimationFrame(() => get('muan-kb-close')?.focus());
    } else if (button.id === 'muan-kb-close') {
      root.dataset.kbOpen = 'false';
      get('muan-kb-open')?.focus();
    } else if (button.id === 'muan-ai-launch') {
      setAssistant(root.dataset.aiOpen !== 'true');
    } else if (button.id === 'muan-ai-close') {
      setAssistant(false);
    } else if (button.id === 'muan-ai-reset') {
      endAssistantGesture();
      placeAssistant(defaultAssistant());
    } else if (button.id === 'muan-enter') {
      root.dataset.muanView = 'workspace';
      setMenu(false);
      syncTab();
      requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
        resizeCharts();
        (narrow() ? get('muan-menu') : selectedTab())?.focus({preventScroll: true});
        window.scrollTo({top: workspaceScroll, behavior: 'instant'});
      });
    } else if (button.id === 'muan-home') {
      // Never hide an active camera and its visual alerts behind the welcome page.
      if (cameraRunning()) {
        get('live-tab-button')?.click();
        announce('实时摄像头仍在使用中，请先在实时监测页停止摄像头，再返回首页。');
        setMenu(false);
        window.scrollTo({top: 0, behavior: 'instant'});
        return;
      }
      workspaceScroll = window.scrollY;
      announce('');
      setMenu(false);
      root.dataset.muanView = 'welcome';
      get('muan-enter')?.focus({preventScroll: true});
      window.scrollTo({top: 0, behavior: 'instant'});
    } else if (button.id === 'muan-menu') {
      const open = root.dataset.muanMenu !== 'open';
      setMenu(open);
      if (open) selectedTab()?.focus({preventScroll: true});
    } else if (button.matches('#workspace-tabs [role="tab"]')) {
      announce('');
      get('muan-current-page').textContent = button.textContent.trim();
      setMenu(false);
      requestAnimationFrame(() => {
        syncTab();
        window.dispatchEvent(new Event('resize'));
        resizeCharts();
        window.scrollTo({top: 0, behavior: 'instant'});
        if (narrow()) get('muan-menu')?.focus({preventScroll: true});
      });
    }
  });

  root.addEventListener('keydown', event => {
    if (!(event.target instanceof Element)) return;
    if (event.target.matches('#muan-ai-drag, #muan-ai-resize') &&
        ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key) && !compactAssistant()) {
      event.preventDefault();
      placeAssistant();
      const step = event.shiftKey ? 5 : 20;
      const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
      const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
      const b = assistantBounds;
      placeAssistant(event.target.id === 'muan-ai-drag' ? {...b, x: b.x + dx, y: b.y + dy} : {...b,
        width: Math.min(b.width + dx, window.innerWidth - b.x - 12),
        height: Math.min(b.height + dy, window.innerHeight - b.y - 12)});
      return;
    }
    if (event.key === 'Escape' && root.dataset.kbOpen === 'true') {
      root.dataset.kbOpen = 'false';
      get('muan-kb-open')?.focus();
      return;
    }
    if (event.key === 'Escape' && root.dataset.aiToolsOpen === 'true') {
      setAssistantTools(false);
      return;
    }
    if (event.key === 'Escape' && root.dataset.aiOpen === 'true') {
      setAssistant(false);
      return;
    }
    if (event.key === 'Escape' && root.dataset.muanMenu === 'open') {
      setMenu(false);
      get('muan-menu')?.focus();
      return;
    }
    if (!event.target.matches('#workspace-tabs [role="tab"]') || !['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const buttons = tabButtons();
    const current = buttons.indexOf(event.target);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
      (current + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
    buttons[next]?.click();
  });
}
