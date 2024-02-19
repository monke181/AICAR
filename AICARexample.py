import pygame
import math
import sys
import neat
import os


pygame.init()

SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("track.png"))
CHECKPOINTTRACK = pygame.image.load(os.path.join("track_with_checkpoints.png"))
#caption for top of screen
pygame.display.set_caption('AICAR')
#score
font = pygame.font.SysFont('Bauhaus 93', 60)
#color
white = (255, 255, 255)
#drawing text
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    SCREEN.blit(img, (x, y))

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.checkpoints = 0
        self.decision_list = []
        
 
    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6
        if self.direction == 2:
            self.brake()



    def brake(self):
        if self.direction == 2:
            self.vel_vector -= pygame.math.Vector2(0.1, 0)
        else:
            if self.direction == 1:
                self.rect.center += self.vel_vector * 6
        




    def collision(self):
        global score
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # die on collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False
        #check if hit checkpoint
        if CHECKPOINTTRACK.get_at(collision_point_right) == pygame.Color(0, 0, 0, 255) \
                or CHECKPOINTTRACK.get_at(collision_point_left) == pygame.Color(0, 0, 0, 255):
            #print("hit Checkpoint")
            score += 1
            self.checkpoints += 1
        

        # draw collision points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # draw radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = float(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])

    def data(self):
        input_data = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input_data[i] = int(radar[1])
        return input_data
    
    

def eval_genomes(genomes, config):
    global cars, ge, nets, score

    cars = []
    ge = []
    nets = []
    score = 0
    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))
        draw_text(str(score), font, white, int(SCREEN_WIDTH /2), 20)
        if len(cars) == 0:
            break

        # Use list comprehension to filter out dead cars
        cars = [car for car in cars if car.sprite.alive]

        for i, car in enumerate(cars):
            ge[i].fitness += 1
        # Direction 0 = forward
        # Direction 2 = slow down
        # Direction 1 = right
        # Direction -1 = left
        for i, car in enumerate(cars):
            output = car.sprite.data()
            # output = []
            # for radar in car.sprite.radars:
            #     output.append[radar[1]]
            print(output)

            if output[0] <= 80 and output[4] <= 150:
                car.sprite.direction = 0
            elif output[2] < 150:
                car.sprite.direction = 2
            elif output[0] < 150 and output[1] < 150:
                car.sprite.direction = -1
                if car.sprite.vel_vector[0] > 1:
                     car.sprite.direction = 2
            elif output[3] < 150 and output[4] < 150:
                car.sprite.direction = 1
                if car.sprite.vel_vector[0] > 1:
                    car.sprite.direction = 2
            
            # turn away from the direction closest to a collision
            # min distance between all the sprite data aside from the middle one
            

        # Update
        for car in cars:
            car.draw(SCREEN)
            car.update()
            # Copy moves of car that got the furthest, pass those moves onto each new car
            # if len(cars) == 1:


        pygame.display.update()
        

def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run for 100 generations
    generations_to_run = 1000
    pop.run(eval_genomes, generations_to_run)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
