// Wall-clock tick placement for chart x-axes, in the viewer's local timezone.
// Ticks snap to calendar boundaries (whole minutes, hours, local midnight,
// month starts) and labels use the same multi-scale scheme as d3's time
// scales: ":30" for seconds, "10:15" for minutes, "10 AM" for hours,
// "Mon 15" for days, "September" for month starts, "2026" for year starts.

const MINUTE = 60;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

// [step seconds, unit]; the unit decides how a tick snaps to the calendar.
const INTERVALS = [
  [1, "second"],
  [5, "second"],
  [15, "second"],
  [30, "second"],
  [MINUTE, "minute"],
  [5 * MINUTE, "minute"],
  [15 * MINUTE, "minute"],
  [30 * MINUTE, "minute"],
  [HOUR, "hour"],
  [3 * HOUR, "hour"],
  [6 * HOUR, "hour"],
  [12 * HOUR, "hour"],
  [DAY, "day"],
  [2 * DAY, "day"],
  [7 * DAY, "week"],
  [30 * DAY, "month"],
  [90 * DAY, "month"],
  [365 * DAY, "year"],
];

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

const pad = (n) => String(n).padStart(2, "0");

function localMidnight(date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

// Sub-day steps count from local midnight so ticks land on local clock
// boundaries even in zones with a fractional UTC offset.
function floorToStep(seconds, step, unit) {
  const date = new Date(seconds * 1000);
  switch (unit) {
    case "second":
    case "minute":
    case "hour": {
      const midnight = localMidnight(date).getTime() / 1000;
      const sinceMidnight = seconds - midnight;
      return midnight + Math.floor(sinceMidnight / step) * step;
    }
    case "day":
      return localMidnight(date).getTime() / 1000;
    case "week": {
      const midnight = localMidnight(date);
      midnight.setDate(midnight.getDate() - midnight.getDay());
      return midnight.getTime() / 1000;
    }
    case "month":
      return new Date(date.getFullYear(), date.getMonth(), 1).getTime() / 1000;
    case "year":
      return new Date(date.getFullYear(), 0, 1).getTime() / 1000;
    default:
      return Math.floor(seconds / step) * step;
  }
}

function advance(seconds, step, unit) {
  const date = new Date(seconds * 1000);
  switch (unit) {
    case "day":
    case "week":
      date.setDate(date.getDate() + Math.round(step / DAY));
      return date.getTime() / 1000;
    case "month":
      date.setMonth(date.getMonth() + Math.max(1, Math.round(step / (30 * DAY))));
      return date.getTime() / 1000;
    case "year":
      date.setFullYear(date.getFullYear() + Math.max(1, Math.round(step / (365 * DAY))));
      return date.getTime() / 1000;
    default:
      return seconds + step;
  }
}

export function pickTimeInterval(duration, count) {
  const target = duration / Math.max(1, count);
  for (const interval of INTERVALS) {
    if (interval[0] >= target) return interval;
  }
  return INTERVALS[INTERVALS.length - 1];
}

/**
 * Tick timestamps (epoch seconds) inside `[start, end]`, aiming for roughly
 * `count` of them.
 */
export function timeTicks(start, end, count) {
  if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) return [];
  const [step, unit] = pickTimeInterval(end - start, count);
  const ticks = [];
  let t = floorToStep(start, step, unit);
  // Guard against a clock that never advances (invalid dates).
  for (let i = 0; i < 1000 && t <= end; i++) {
    if (t >= start) ticks.push(t);
    const next = advance(t, step, unit);
    if (!(next > t)) break;
    t = next;
  }
  return ticks;
}

export function formatTimeTick(seconds) {
  const d = new Date(seconds * 1000);
  if (Number.isNaN(d.getTime())) return "";
  const hours = d.getHours();
  const hour12 = hours % 12 === 0 ? 12 : hours % 12;
  const meridiem = hours < 12 ? "AM" : "PM";
  if (d.getSeconds()) return `:${pad(d.getSeconds())}`;
  if (d.getMinutes()) return `${pad(hour12)}:${pad(d.getMinutes())}`;
  if (hours) return `${pad(hour12)} ${meridiem}`;
  if (d.getDate() !== 1) return `${DAYS[d.getDay()]} ${pad(d.getDate())}`;
  if (d.getMonth()) return MONTHS[d.getMonth()];
  return String(d.getFullYear());
}
