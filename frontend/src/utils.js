export const GRID_SIZE = 20;

export function snapToGrid(value, gridSize = GRID_SIZE) {
  return Math.round(value / gridSize) * gridSize;
}

export function snapPoint(point, gridSize = GRID_SIZE) {
  return {
    x: snapToGrid(point.x, gridSize),
    y: snapToGrid(point.y, gridSize)
  };
}

export function wallLength(wall) {
  const dx = wall.end.x - wall.start.x;
  const dy = wall.end.y - wall.start.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function computeWallDelta(previous, next) {
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
