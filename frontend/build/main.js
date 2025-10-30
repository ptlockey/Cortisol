(() => {
  const { useState, useEffect, useMemo, useCallback, useRef } = React;
  const { Stage, Layer, Line, Circle, Text } = ReactKonva;

  const GRID_SIZE = 20;

  const DEFAULT_DATA = {
    version: "1.0.0",
    walls: [],
    rooms: [],
    metadata: {
      createdWith: "floorplan-component"
    }
  };

  function snapToGrid(value, gridSize = GRID_SIZE) {
    return Math.round(value / gridSize) * gridSize;
  }

  function snapPoint(point, gridSize = GRID_SIZE) {
    return {
      x: snapToGrid(point.x, gridSize),
      y: snapToGrid(point.y, gridSize)
    };
  }

  function wallLength(wall) {
    const dx = wall.end.x - wall.start.x;
    const dy = wall.end.y - wall.start.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function computeWallDelta(previous, next) {
    const prevMap = new Map(previous.map((wall) => [wall.id, wall]));
    const nextMap = new Map(next.map((wall) => [wall.id, wall]));

    const added = [];
    const updated = [];
    const removed = [];

    nextMap.forEach((wall, id) => {
      if (!prevMap.has(id)) {
        added.push(id);
        return;
      }
      const prev = prevMap.get(id);
      if (JSON.stringify(prev) !== JSON.stringify(wall)) {
        updated.push(id);
      }
    });

    prevMap.forEach((_wall, id) => {
      if (!nextMap.has(id)) {
        removed.push(id);
      }
    });

    return { addedWalls: added, updatedWalls: updated, removedWalls: removed };
  }

  const ToolbarButton = ({ active, onClick, children, disabled }) =>
    React.createElement(
      "button",
      {
        type: "button",
        onClick,
        className: active ? "active" : undefined,
        disabled
      },
      children
    );

  const WallEditorCanvas = ({ data, width, height, disabled, onChange }) => {
    const [walls, setWalls] = useState(() => deepClone(data?.walls ?? []));
    const [rooms, setRooms] = useState(() => deepClone(data?.rooms ?? []));
    const [mode, setMode] = useState("select");
    const [snapEnabled, setSnapEnabled] = useState(true);
    const [draftStart, setDraftStart] = useState(null);
    const [draftEnd, setDraftEnd] = useState(null);
    const [selectedWall, setSelectedWall] = useState(null);
    const stageRef = useRef(null);
    const previousWallsRef = useRef(deepClone(data?.walls ?? []));

    useEffect(() => {
      const nextWalls = deepClone(data?.walls ?? []);
      setWalls(nextWalls);
      setRooms(deepClone(data?.rooms ?? []));
      previousWallsRef.current = deepClone(nextWalls);
      setSelectedWall(null);
      setDraftStart(null);
      setDraftEnd(null);
    }, [JSON.stringify(data)]);

    useEffect(() => {
      sendFrameHeight(height + 80);
    }, [height, walls.length]);

    const effectiveWidth = Math.max(640, (width ?? 0) - 24);
    const effectiveHeight = Math.max(480, height - 24);

    const gridLines = useMemo(() => {
      const vertical = [];
      const horizontal = [];
      for (let x = 0; x <= effectiveWidth; x += GRID_SIZE) {
        vertical.push(x);
      }
      for (let y = 0; y <= effectiveHeight; y += GRID_SIZE) {
        horizontal.push(y);
      }
      return { vertical, horizontal };
    }, [effectiveWidth, effectiveHeight]);

    const emitChange = useCallback(
      (nextWalls) => {
        const nextData = {
          ...data,
          walls: deepClone(nextWalls),
          rooms: deepClone(rooms)
        };
        const delta = computeWallDelta(previousWallsRef.current, nextWalls);
        previousWallsRef.current = deepClone(nextWalls);
        onChange({
          data: nextData,
          delta: {
            ...delta,
            timestamp: new Date().toISOString()
          }
        });
      },
      [data, onChange, rooms]
    );

    const handleModeChange = (nextMode) => () => {
      setMode(nextMode);
      setDraftStart(null);
      setDraftEnd(null);
    };

    const handleToggleSnap = () => setSnapEnabled((value) => !value);

    const handleStageClick = () => {
      if (disabled || mode !== "draw") {
        return;
      }
      const stage = stageRef.current;
      const pointer = stage?.getPointerPosition?.();
      if (!pointer) {
        return;
      }
      const adjusted = snapEnabled ? snapPoint(pointer) : pointer;
      if (!draftStart) {
        setDraftStart(adjusted);
        setDraftEnd(adjusted);
      } else {
        const id = `wall-${Date.now()}`;
        const endPoint = draftEnd ?? adjusted;
        const newWall = {
          id,
          start: draftStart,
          end: endPoint,
          thickness: 10
        };
        const nextWalls = [...walls, newWall];
        setWalls(nextWalls);
        setDraftStart(null);
        setDraftEnd(null);
        emitChange(nextWalls);
        setSelectedWall(id);
      }
    };

    const handleStageMouseMove = () => {
      if (disabled || mode !== "draw" || !draftStart) {
        return;
      }
      const stage = stageRef.current;
      const pointer = stage?.getPointerPosition?.();
      if (!pointer) {
        return;
      }
      const adjusted = snapEnabled ? snapPoint(pointer) : pointer;
      setDraftEnd(adjusted);
    };

    const handleWallClick = (wallId) => (event) => {
      if (disabled) {
        return;
      }
      if (mode === "delete") {
        const nextWalls = walls.filter((wall) => wall.id !== wallId);
        setWalls(nextWalls);
        emitChange(nextWalls);
        setSelectedWall(null);
        if (event?.cancelBubble !== undefined) {
          event.cancelBubble = true;
        }
        return;
      }
      setSelectedWall(wallId);
    };

    const updateWallAnchor = (wallId, anchor, commit) => (event) => {
      if (disabled) {
        return;
      }
      const position = event.target.position();
      const point = snapEnabled ? snapPoint(position) : position;
      event.target.position(point);
      setWalls((previousWalls) => {
        const nextWalls = previousWalls.map((wall) =>
          wall.id === wallId ? { ...wall, [anchor]: { x: point.x, y: point.y } } : wall
        );
        if (commit) {
          emitChange(nextWalls);
        }
        return nextWalls;
      });
    };

    const wallAnnotations = useMemo(
      () =>
        walls.map((wall) => {
          const centreX = (wall.start.x + wall.end.x) / 2;
          const centreY = (wall.start.y + wall.end.y) / 2;
          const length = wallLength(wall).toFixed(1);
          return { id: wall.id, x: centreX, y: centreY, length };
        }),
      [walls]
    );

    const draftLinePoints = draftStart && draftEnd
      ? [draftStart.x, draftStart.y, draftEnd.x, draftEnd.y]
      : [];

    const toolbarMessage = useMemo(() => {
      if (disabled) {
        return "Read-only mode";
      }
      if (mode === "draw") {
        return draftStart ? "Click to finish wall" : "Click to place the first point";
      }
      if (mode === "delete") {
        return "Click a wall to remove it";
      }
      return selectedWall ? `Selected wall: ${selectedWall}` : "Select a wall to inspect";
    }, [mode, draftStart, disabled, selectedWall]);

    return React.createElement(
      "div",
      { className: "editor-container" },
      React.createElement(
        "div",
        { className: "toolbar" },
        React.createElement(ToolbarButton, {
          active: mode === "select",
          onClick: handleModeChange("select"),
          disabled
        }, "Select"),
        React.createElement(ToolbarButton, {
          active: mode === "draw",
          onClick: handleModeChange("draw"),
          disabled
        }, "Draw wall"),
        React.createElement(ToolbarButton, {
          active: mode === "delete",
          onClick: handleModeChange("delete"),
          disabled
        }, "Delete wall"),
        React.createElement(ToolbarButton, {
          onClick: handleToggleSnap,
          active: snapEnabled,
          disabled
        }, "Snap to grid")
      ),
      React.createElement("div", { className: "annotation" }, toolbarMessage),
      React.createElement(
        "div",
        { className: "canvas-wrapper" },
        React.createElement(
          Stage,
          {
            ref: stageRef,
            width: effectiveWidth,
            height: effectiveHeight,
            onMouseDown: handleStageClick,
            onMouseMove: handleStageMouseMove,
            listening: !disabled,
            style: { cursor: mode === "draw" ? "crosshair" : "default" }
          },
          React.createElement(
            Layer,
            null,
            gridLines.vertical.map((x) =>
              React.createElement(Line, {
                key: `grid-x-${x}`,
                points: [x, 0, x, effectiveHeight],
                stroke: "#e5e5e5",
                strokeWidth: 1
              })
            ),
            gridLines.horizontal.map((y) =>
              React.createElement(Line, {
                key: `grid-y-${y}`,
                points: [0, y, effectiveWidth, y],
                stroke: "#e5e5e5",
                strokeWidth: 1
              })
            )
          ),
          React.createElement(
            Layer,
            null,
            walls.map((wall) =>
              React.createElement(
                React.Fragment,
                { key: wall.id },
                React.createElement(Line, {
                  points: [wall.start.x, wall.start.y, wall.end.x, wall.end.y],
                  stroke: selectedWall === wall.id ? "#ff4081" : "#1f2937",
                  strokeWidth: wall.thickness / 2,
                  onClick: handleWallClick(wall.id)
                }),
                React.createElement(Circle, {
                  x: wall.start.x,
                  y: wall.start.y,
                  radius: 6,
                  fill: selectedWall === wall.id ? "#ff4081" : "#2563eb",
                  draggable: !disabled,
                  dragBoundFunc: (pos) => (snapEnabled ? snapPoint(pos) : pos),
                  onDragMove: updateWallAnchor(wall.id, "start", false),
                  onDragEnd: updateWallAnchor(wall.id, "start", true)
                }),
                React.createElement(Circle, {
                  x: wall.end.x,
                  y: wall.end.y,
                  radius: 6,
                  fill: selectedWall === wall.id ? "#ff4081" : "#2563eb",
                  draggable: !disabled,
                  dragBoundFunc: (pos) => (snapEnabled ? snapPoint(pos) : pos),
                  onDragMove: updateWallAnchor(wall.id, "end", false),
                  onDragEnd: updateWallAnchor(wall.id, "end", true)
                })
              )
            ),
            draftLinePoints.length === 4
              ? React.createElement(Line, {
                  points: draftLinePoints,
                  stroke: "#0f172a",
                  dash: [8, 4],
                  strokeWidth: 3
                })
              : null
          ),
          React.createElement(
            Layer,
            null,
            wallAnnotations.map((annotation) =>
              React.createElement(Text, {
                key: `annotation-${annotation.id}`,
                x: annotation.x + 6,
                y: annotation.y + 6,
                text: `${annotation.length} px`,
                fontSize: 12,
                fill: "#475569"
              })
            )
          )
        )
      )
    );
  };

  const FloorplanApp = ({ args }) => {
    const data = args?.data ?? DEFAULT_DATA;
    const disabled = args?.disabled ?? false;
    const height = args?.height ?? 720;
    const width = args?.width ?? document.documentElement.clientWidth ?? 800;

    const handleChange = useCallback((payload) => {
      setComponentValue(payload);
    }, []);

    return React.createElement(WallEditorCanvas, {
      data,
      width,
      height,
      disabled,
      onChange: handleChange
    });
  };

  let root;

  function renderComponent(renderArgs) {
    if (!root) {
      const container = document.getElementById("root");
      root = ReactDOM.createRoot(container);
    }
    root.render(React.createElement(FloorplanApp, { args: renderArgs }));
  }

  function sendMessage(type, payload) {
    window.parent.postMessage({ type, ...payload }, "*");
  }

  function setComponentReady() {
    sendMessage("streamlit:componentReady", {});
  }

  function sendFrameHeight(height) {
    sendMessage("streamlit:setFrameHeight", { height });
  }

  function setComponentValue(value) {
    sendMessage("streamlit:setComponentValue", { value });
  }

  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || data.type !== "streamlit:render") {
      return;
    }
    const width = data.args?.width ?? data.width ?? data.bounds?.width ?? document.documentElement.clientWidth ?? 800;
    const height = data.args?.height ?? 720;
    const args = {
      ...data.args,
      disabled: data.disabled,
      theme: data.theme,
      width,
      height
    };
    renderComponent(args);
  });

  setComponentReady();
})();
