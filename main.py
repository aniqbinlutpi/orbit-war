"""
Orbit Wars heuristic agent.

This version is a stronger baseline than the nearest-planet demo:
it values targets by production, estimates ships at arrival time,
keeps a defense reserve at home, and avoids launches that cut
through the sun.
"""

import math
from collections import namedtuple

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
except ImportError:
    Planet = namedtuple(
        "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
    )


CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_MOVES_PER_TURN = 12


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


def future_target_ships(target, eta):
    if target.owner == -1:
        return target.ships
    return target.ships + target.production * eta


def local_threat(source, enemy_planets):
    threat = 0.0
    for enemy in enemy_planets:
        dist = distance(source, enemy)
        if dist > 35:
            continue
        threat += max(0.0, enemy.ships - source.production) / max(6.0, dist)
    return threat


def reserve_ships(source, enemy_planets):
    base = 8 + 2 * source.production
    if distance(source, Planet(-1, -1, CENTER_X, CENTER_Y, 0, 0, 0)) < 24:
        base += 4
    return min(source.ships, math.ceil(base + local_threat(source, enemy_planets)))


def target_score(source, target, eta, ships_needed):
    dist = distance(source, target)
    production_value = target.production * (5.5 if target.owner == -1 else 7.0)
    capture_discount = ships_needed * 1.25
    distance_discount = dist * 0.35 + eta * 0.8
    ownership_bonus = 7.0 if target.owner == -1 else 10.0
    snowball_bonus = max(0.0, 16.0 - target.ships * 0.4)
    return production_value + ownership_bonus + snowball_bonus - capture_discount - distance_discount


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*planet) for planet in raw_planets]

    my_planets = [planet for planet in planets if planet.owner == player]
    enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
    targets = [planet for planet in planets if planet.owner != player]

    if not my_planets or not targets:
        return []

    outgoing_commitments = {planet.id: 0 for planet in my_planets}
    committed_to_target = {planet.id: 0 for planet in targets}
    candidate_moves = []

    for source in my_planets:
        reserve = reserve_ships(source, enemy_planets)
        available = source.ships - reserve
        if available <= 0:
            continue

        best_option = None
        for target in targets:
            if source.id == target.id or line_hits_sun(source, target):
                continue

            probe_size = max(1, min(available, 60))
            eta = travel_time(source, target, probe_size)
            future_ships = future_target_ships(target, eta)
            already_sent = committed_to_target.get(target.id, 0)
            ships_needed = max(1, future_ships + 1 - already_sent)

            if ships_needed > available:
                continue

            score = target_score(source, target, eta, ships_needed)
            if score < 2.0:
                continue

            if best_option is None or score > best_option["score"]:
                best_option = {
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
        reserve = reserve_ships(source, enemy_planets)
        available = source.ships - reserve - outgoing_commitments[source.id]
        ships = min(plan["ships"], available)
        if ships <= 0:
            continue

        angle = math.atan2(target.y - source.y, target.x - source.x)
        moves.append([source.id, angle, ships])
        outgoing_commitments[source.id] += ships
        committed_to_target[target.id] = committed_to_target.get(target.id, 0) + ships

    return moves
