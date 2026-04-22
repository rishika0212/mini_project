import carla
import math

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
lights = world.get_actors().filter('traffic.traffic_light')

groups = []
for l in lights:
    loc = l.get_location()
    added = False
    for g in groups:
        if loc.distance(g[0].get_location()) < 45:
            g.append(l)
            added = True
            break
    if not added:
        groups.append([l])

groups.sort(key=lambda g: sum(l.get_location().distance(carla.Location(0,0,0)) for l in g)/len(g))
g = groups[0]
cx = sum(l.get_location().x for l in g)/len(g)
cy = sum(l.get_location().y for l in g)/len(g)

print(f"Center: {cx:.1f}, {cy:.1f}")
for l in g:
    dx = l.get_location().x - cx
    dy = l.get_location().y - cy
    angle = math.degrees(math.atan2(dy, dx))
    print(f"Light {l.id} at dx={dx:.1f}, dy={dy:.1f}, angle={angle:.1f}")
