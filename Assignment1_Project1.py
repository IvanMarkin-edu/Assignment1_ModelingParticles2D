import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


#  ласс частицы
class Particle:
    def __init__(self, x, y, vx, vy, mass, radius):
        self.x = x                    # координата x частицы
        self.y = y                    # координата y частицы
        self.vx = vx                  # y-компонента скорости
        self.vy = vy                  # x-компонента скорости частицы
        self.mass = mass              # масса частицы
        self.radius = radius          # радиус частицы

# —оздание случайных частиц
def create_particles(num_particles, box_size):
    particles = []
    for _ in range(num_particles):
        while True:
            # ¬озможные параметры
            x = np.random.uniform(0.2, box_size - 0.2)
            y = np.random.uniform(0.2, box_size - 0.2)
            radius = 0.3

            # ѕроверка, что частицы не сталкиваютс€ при генерации
            collision = False
            for p in particles:
                dx = x - p.x
                dy = y - p.y
                if np.sqrt(dx**2 + dy**2) < radius + p.radius:
                    collision = True
                    break
            if not collision:
                break

        vx = np.random.uniform(-10, 10)
        vy = np.random.uniform(-10, 10)
        mass = 1.0
        particles.append(Particle(x, y, vx, vy, mass, radius))
    return particles

# ƒвижение частиц
def regular_movement(particles, dt):
    for p in particles:
        p.x += p.vx * dt
        p.y += p.vy * dt

# ќбработка столкновени€ со стеной
def dealwith_wall_collisions(particles, box_size):
    for p in particles:
        if p.x < p.radius:
            p.x = p.radius
            p.vx = abs(p.vx)
        elif p.x > box_size - p.radius:
            p.x = box_size - p.radius
            p.vx = -abs(p.vx)
        if p.y < p.radius:
            p.y = p.radius
            p.vy = abs(p.vy)
        elif p.y > box_size - p.radius:
            p.y = box_size - p.radius
            p.vy = -abs(p.vy)

# ќбработка столкновени€ частиц
def dealwith_particle_collisions(particles):
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = particles[i]
            p2 = particles[j]
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            distance_sq = dx**2 + dy**2
            min_dist = p1.radius + p2.radius
            
            if distance_sq < min_dist**2:
                distance = np.sqrt(distance_sq)
                # вектор нормали, направленный от центра одной частицы к центру другой
                nx = dx / distance
                ny = dy / distance
 
                # относительна€ скорость в проекции на вектор нормали
                dvx = p1.vx - p2.vx
                dvy = p1.vy - p2.vy
                velocity_normal = dvx * nx + dvy * ny
                
                # если частицы движутс€ друг от друга, то столкновени€ нет
                if velocity_normal > 0:
                    continue
                
                c = -(2.0) * velocity_normal / (1/p1.mass + 1/p2.mass)
                
                # обновление скоростей
                p1.vx += c * nx / p1.mass
                p1.vy += c * ny / p1.mass
                p2.vx -= c * nx / p2.mass
                p2.vy -= c * ny / p2.mass
                
                # исправление пересечений частиц
                overlap = (min_dist - distance) / 2.0
                p1.x += nx * overlap
                p1.y += ny * overlap
                p2.x -= nx * overlap
                p2.y -= ny * overlap

def simulate(num_particles, box_size=5, dt=0.01, total_time=1):
    particles = create_particles(num_particles, box_size)
    data = []
    num_steps = int(total_time / dt)
    
    for _ in range(num_steps):
        current_state = []
        for p in particles:
            current_state.append({
                'x': p.x,
                'y': p.y,
                'vx': p.vx,
                'vy': p.vy,
                'mass': p.mass,
                'radius': p.radius
            })
        data.append(current_state)
        
        regular_movement(particles, dt)
        dealwith_wall_collisions(particles, box_size)
        dealwith_particle_collisions(particles)
    
    return data, box_size

def animate_simulation(data, box_size):
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    
    circles = []
    for p in data[0]:
        circle = plt.Circle((p['x'], p['y']), p['radius'], fc='blue')
        ax.add_patch(circle)
        circles.append(circle)
    
    def animate(frame):
        for i, circle in enumerate(circles):
            pos = frame[i]
            circle.center = (pos['x'], pos['y'])
        return circles
    
    ani = animation.FuncAnimation(fig, animate, frames=data, interval=50, blit=True)
    ani.save("gif7.gif", writer="pillow")
    plt.show()

if __name__ == "__main__":
    num_particles = 10

    # вывод анимации
    simulation_data, box_size = simulate(num_particles=num_particles)
    animate_simulation(simulation_data, box_size)


    # Ќабор датасета.
#    start_time = time.time()
#    for _ in range(10000):
#        simulation_data, box_size = simulate(num_particles=num_particles)
#
#        newstr = ''
#        with open("Dataset10000.txt", 'a') as file:
#            for i in range(10):
#                for value in simulation_data[0][i].values():
#                    newstr += str(value) 
#                    newstr += ' '
#            for i in range(10):
#                for value in simulation_data[-1][i].values():
#                    newstr += str(value) 
#                    newstr += ' '
#                newstr = newstr[:-8]
#            file.write(f"{newstr}\n")
#            newstr = ''
#    end_time = time.time()
#    print(end_time - start_time)

