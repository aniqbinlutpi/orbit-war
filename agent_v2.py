"""
Orbit Wars heuristic agent.

This version goes beyond simple expansion:
- estimates future garrisons
- preserves defense against nearby threats
- reinforces front-line planets from safer back-line planets
- becomes more aggressive once it is ahead
"""

import math
from collections import namedtuple

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
except ImportError:
    Planet = namedtuple(
        "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
    )
    Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_MOVES_PER_TURN = 14
FRONTLINE_RANGE = 28.0


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fleet_speed(ships):
    ships = max(1, ships)
    if ships == 1:
        return 1.0
    scale = math.log(ships) / math.log(1000)
    return 1.0 + 5.0 * (scale**1.5)


def travel_time(source, target, ships):
    path = max(0.0, distance(source, target) - source.radius - target.radius)
    return max(1, math.ceil(path / fleet_speed(ships)))


def line_hits_sun(source, target, margin=0.6):
    dx = target.x - source.x
    dy = target.y - source.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return False

    t = ((CENTER_X - source.x) * dx + (CENTER_Y - source.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = source.x + t * dx
    closest_y = source.y + t * dy
    return math.hypot(closest_x - CENTER_X, closest_y - CENTER_Y) <= SUN_RADIUS + margin


def nearest_planet_to_fleet(fleet, planets):
    best = None
    best_dist = float("inf")
    for planet in planets:
        dist = math.hypot(fleet.x - planet.x, fleet.y - planet.y)
        if dist < best_dist:
            best_dist = dist
            best = planet
    return best, best_dist


def future_target_ships(target, eta):
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta


def incoming_fleet_pressure(my_planets, enemy_fleets, allied_fleets):
    enemy_pressure = {planet.id: 0 for planet in my_planets}
    allied_pressure = {planet.id: 0 for planet in my_planets}

    for fleet in enemy_fleets:
        planet, dist = nearest_planet_to_fleet(fleet, my_planets)
        if planet is not None and dist <= 20:
            enemy_pressure[planet.id] += fleet.ships

    for fleet in allied_fleets:
        planet, dist = nearest_planet_to_fleet(fleet, my_planets)
        if planet is not None and dist <= 18:
            allied_pressure[planet.id] += fleet.ships

    return enemy_pressure, allied_pressure


def local_threat(source, enemy_planets, enemy_pressure):
    threat = enemy_pressure.get(source.id, 0) * 0.8
    for enemy in enemy_planets:
        dist = distance(source, enemy)
        if dist > 36:
            continue
        threat += max(0.0, enemy.ships + enemy.production * 2 - source.production) / max(6.0, dist)
    return threat


def reserve_ships(source, enemy_planets, enemy_pressure, allied_pressure):
    base = 8 + 2 * source.production
    if math.hypot(source.x - CENTER_X, source.y - CENTER_Y) < 24:
        base += 4
    reserve = base + local_threat(source, enemy_planets, enemy_pressure)
    reserve -= allied_pressure.get(source.id, 0) * 0.35
    return max(0, min(source.ships, math.ceil(reserve)))


def front_line_score(source, enemy_planets):
    if not enemy_planets:
        return 0.0
    nearest_enemy = min(distance(source, enemy) for enemy in enemy_planets)
    centrality = 28.0 - math.hypot(source.x - CENTER_X, source.y - CENTER_Y) * 0.3
    return centrality + max(0.0, FRONTLINE_RANGE - nearest_enemy)


def choose_mode(my_planets, enemy_planets, my_fleets, enemy_fleets):
    my_total = sum(planet.ships + 3 * planet.production for planet in my_planets) + sum(
        fleet.ships for fleet in my_fleets
    )
    enemy_total = sum(
        planet.ships + 3 * planet.production for planet in enemy_planets
    ) + sum(fleet.ships for fleet in enemy_fleets)

    if enemy_total == 0:
        return "ahead"
    if my_total > enemy_total * 1.18:
        return "ahead"
    if enemy_total > my_total * 1.08:
        return "defensive"
    return "balanced"


def target_score(source, target, eta, ships_needed, mode):
    dist = distance(source, target)
    production_value = target.production * (6.0 if target.owner == -1 else 8.5)
    capture_discount = ships_needed * 1.2
    distance_discount = dist * 0.32 + eta * 0.85
    ownership_bonus = 8.0 if target.owner == -1 else 12.0
    snowball_bonus = max(0.0, 18.0 - target.ships * 0.35)

    if mode == "ahead" and target.owner != -1:
        ownership_bonus += 6.0
        production_value += 2.0
    if mode == "defensive" and target.owner == -1:
        ownership_bonus -= 3.0

    return production_value + ownership_bonus + snowball_bonus - capture_discount - distance_discount


def weakest_enemy(enemy_planets):
    if not enemy_planets:
        return None

    by_owner = {}
    for planet in enemy_planets:
        by_owner.setdefault(planet.owner, 0)
        by_owner[planet.owner] += planet.ships + 2 * planet.production
    return min(by_owner, key=by_owner.get)


def build_attack_moves(
    my_planets,
    enemy_planets,
    targets,
    enemy_pressure,
    allied_pressure,
    mode,
):
    outgoing_commitments = {planet.id: 0 for planet in my_planets}
    committed_to_target = {planet.id: 0 for planet in targets}
    candidate_moves = []
    focus_owner = weakest_enemy(enemy_planets) if mode == "ahead" else None

    for source in my_planets:
        reserve = reserve_ships(source, enemy_planets, enemy_pressure, allied_pressure)
        available = source.ships - reserve
        if available <= 0:
            continue

        best_option = None
        for target in targets:
            if source.id == target.id or line_hits_sun(source, target):
                continue
            if focus_owner is not None and target.owner not in (-1, focus_owner):
                continue

            probe_size = max(1, min(available, 80))
            eta = travel_time(source, target, probe_size)
            future_ships = future_target_ships(target, eta)
            already_sent = committed_to_target.get(target.id, 0)
            ships_needed = max(1, future_ships + 1 - already_sent)
            if target.owner == -1 and eta > 18:
                ships_needed += 2

            if ships_needed > available:
                continue

            score = target_score(source, target, eta, ships_needed, mode)
            if score < 2.0:
                continue

            if best_option is None or score > best_option["score"]:
                best_option = {
                    "kind": "attack",
                    "source": source,
                    "target": target,
                    "ships": ships_needed,
                    "score": score,
                }

        if best_option is not None:
            candidate_moves.append(best_option)

    candidate_moves.sort(key=lambda move: move["score"], reverse=True)
    moves = []
    for plan in candidate_moves:
        if len(moves) >= MAX_MOVES_PER_TURN:
            break

        source = plan["source"]
        target = plan["target"]
        reserve = reserve_ships(source, enemy_planets, enemy_pressure, allied_pressure)
        available = source.ships - reserve - outgoing_commitments[source.id]
        ships = min(plan["ships"], available)
        if ships <= 0:
            continue

        angle = math.atan2(target.y - source.y, target.x - source.x)
        moves.append([source.id, angle, ships])
        outgoing_commitments[source.id] += ships
        committed_to_target[target.id] = committed_to_target.get(target.id, 0) + ships

    return moves, outgoing_commitments


def build_reinforcement_moves(
    my_planets,
    enemy_planets,
    enemy_pressure,
    allied_pressure,
    outgoing_commitments,
):
    moves = []
    frontliners = sorted(
        my_planets, key=lambda planet: front_line_score(planet, enemy_planets), reverse=True
    )
    if not frontliners:
        return moves

    urgent_targets = [
        planet
        for planet in frontliners
        if enemy_pressure.get(planet.id, 0) > planet.ships * 0.45
        or front_line_score(planet, enemy_planets) > 15
    ]
    if not urgent_targets:
        urgent_targets = frontliners[:2]

    for source in sorted(my_planets, key=lambda planet: front_line_score(planet, enemy_planets)):
        reserve = reserve_ships(source, enemy_planets, enemy_pressure, allied_pressure)
        available = source.ships - reserve - outgoing_commitments.get(source.id, 0)
        if available <= 6:
            continue
        if front_line_score(source, enemy_planets) > 12:
            continue

        best_target = None
        best_score = float("-inf")
        for target in urgent_targets:
            if source.id == target.id or line_hits_sun(source, target):
                continue
            eta = travel_time(source, target, max(1, min(available, 80)))
            score = front_line_score(target, enemy_planets) - eta * 0.9
            score += enemy_pressure.get(target.id, 0) * 0.08
            if score > best_score:
                best_score = score
                best_target = target

        if best_target is None or best_score < 2.0:
            continue

        ships = max(0, int(available * 0.6))
        if ships <= 0:
            continue

        angle = math.atan2(best_target.y - source.y, best_target.x - source.x)
        moves.append([source.id, angle, ships])
        outgoing_commitments[source.id] = outgoing_commitments.get(source.id, 0) + ships

        if len(moves) >= max(3, MAX_MOVES_PER_TURN // 3):
            break

    return moves


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else obs.fleets
    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]

    my_planets = [planet for planet in planets if planet.owner == player]
    enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
    targets = [planet for planet in planets if planet.owner != player]
    my_fleets = [fleet for fleet in fleets if fleet.owner == player]
    enemy_fleets = [fleet for fleet in fleets if fleet.owner not in (-1, player)]

    if not my_planets or not targets:
        return []

    allied_pressure = {planet.id: 0 for planet in my_planets}
    enemy_pressure, allied_pressure = incoming_fleet_pressure(
        my_planets, enemy_fleets, my_fleets
    )
    mode = choose_mode(my_planets, enemy_planets, my_fleets, enemy_fleets)

    attack_moves, outgoing_commitments = build_attack_moves(
        my_planets,
        enemy_planets,
        targets,
        enemy_pressure,
        allied_pressure,
        mode,
    )
    reinforce_moves = build_reinforcement_moves(
        my_planets,
        enemy_planets,
        enemy_pressure,
        allied_pressure,
        outgoing_commitments,
    )

    moves = attack_moves + reinforce_moves
    return moves[:MAX_MOVES_PER_TURN]
