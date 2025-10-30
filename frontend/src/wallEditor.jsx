import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Stage, Layer, Line, Circle, Text } from "react-konva";
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection
} from "streamlit-component-lib";

import { GRID_SIZE, computeWallDelta, deepClone, snapPoint, wallLength } from "./utils";

const DEFAULT_DATA = {
  version: "1.0.0",
  walls: [],
  rooms: [],
  metadata: {
    createdWith: "floorplan-component"
  }
};

const ToolbarButton = ({ active, onClick, children, disabled }) => (
  <button
    type="button"
    onClick={onClick}
    className={active ? "active" : undefined}
    disabled={disabled}
  >
    {children}
  </button>
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

  const effectiveWidth = Math.max(640, width - 24);
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
      event.cancelBubble = true;
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

  const wallAnnotations = useMemo(() =>
    walls.map((wall) => {
      const centreX = (wall.start.x + wall.end.x) / 2;
      const centreY = (wall.start.y + wall.end.y) / 2;
      const length = wallLength(wall).toFixed(1);
      return { id: wall.id, x: centreX, y: centreY, length };
    }),
  [walls]);

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

  return (
    <div className="editor-container">
      <div className="toolbar">
        <ToolbarButton active={mode === "select"} onClick={handleModeChange("select")} disabled={disabled}>
          Select
        </ToolbarButton>
        <ToolbarButton active={mode === "draw"} onClick={handleModeChange("draw")} disabled={disabled}>
          Draw wall
        </ToolbarButton>
        <ToolbarButton active={mode === "delete"} onClick={handleModeChange("delete")} disabled={disabled}>
          Delete wall
        </ToolbarButton>
        <ToolbarButton onClick={handleToggleSnap} active={snapEnabled} disabled={disabled}>
          Snap to grid
        </ToolbarButton>
      </div>
      <div className="annotation">{toolbarMessage}</div>
      <div className="canvas-wrapper">
        <Stage
          ref={stageRef}
          width={effectiveWidth}
          height={effectiveHeight}
          onMouseDown={handleStageClick}
          onMouseMove={handleStageMouseMove}
          listening={!disabled}
          style={{ cursor: mode === "draw" ? "crosshair" : "default" }}
        >
          <Layer>
            {gridLines.vertical.map((x) => (
              <Line key={`grid-x-${x}`} points={[x, 0, x, effectiveHeight]} stroke="#e5e5e5" strokeWidth={1} />
            ))}
            {gridLines.horizontal.map((y) => (
              <Line key={`grid-y-${y}`} points={[0, y, effectiveWidth, y]} stroke="#e5e5e5" strokeWidth={1} />
            ))}
          </Layer>
          <Layer>
            {walls.map((wall) => (
              <React.Fragment key={wall.id}>
                <Line
                  points={[wall.start.x, wall.start.y, wall.end.x, wall.end.y]}
                  stroke={selectedWall === wall.id ? "#ff4081" : "#1f2937"}
                  strokeWidth={wall.thickness / 2}
                  onClick={handleWallClick(wall.id)}
                />
                <Circle
                  x={wall.start.x}
                  y={wall.start.y}
                  radius={6}
                  fill={selectedWall === wall.id ? "#ff4081" : "#2563eb"}
                  draggable={!disabled}
                  dragBoundFunc={(pos) => (snapEnabled ? snapPoint(pos) : pos)}
                  onDragMove={updateWallAnchor(wall.id, "start", false)}
                  onDragEnd={updateWallAnchor(wall.id, "start", true)}
                />
                <Circle
                  x={wall.end.x}
                  y={wall.end.y}
                  radius={6}
                  fill={selectedWall === wall.id ? "#ff4081" : "#2563eb"}
                  draggable={!disabled}
                  dragBoundFunc={(pos) => (snapEnabled ? snapPoint(pos) : pos)}
                  onDragMove={updateWallAnchor(wall.id, "end", false)}
                  onDragEnd={updateWallAnchor(wall.id, "end", true)}
                />
              </React.Fragment>
            ))}
            {draftLinePoints.length === 4 && (
              <Line
                points={draftLinePoints}
                stroke="#0f172a"
                dash={[8, 4]}
                strokeWidth={3}
              />
            )}
          </Layer>
          <Layer>
            {wallAnnotations.map((annotation) => (
              <Text
                key={`annotation-${annotation.id}`}
                x={annotation.x + 6}
                y={annotation.y + 6}
                text={`${annotation.length} px`}
                fontSize={12}
                fill="#475569"
              />
            ))}
          </Layer>
        </Stage>
      </div>
    </div>
  );
};

class FloorplanComponent extends StreamlitComponentBase {
  componentDidMount() {
    Streamlit.setFrameHeight(760);
  }

  componentDidUpdate() {
    Streamlit.setFrameHeight();
  }

  handleChange = (payload) => {
    this.setComponentValue(payload);
  };

  render() {
    const { data = DEFAULT_DATA, disabled = false, height = 720 } = this.props.args;
    const width = this.props.width ?? 720;
    return (
      <WallEditorCanvas
        data={data ?? DEFAULT_DATA}
        width={width}
        height={height}
        disabled={disabled}
        onChange={this.handleChange}
      />
    );
  }
}

export default withStreamlitConnection(FloorplanComponent);
