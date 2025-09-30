import pygame
import random
import numpy as np

pygame.init()

COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'gold': (255, 215, 0)
}

WIDTH, HEIGHT = 800, 600
SNAKE_SIZE = 20
SNAKE_SPEED = 15

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake RL Game")

clock = pygame.time.Clock()


class Snake:
    def __init__(self):
        self.size = SNAKE_SIZE
        self.speed = SNAKE_SPEED
        self.reset()

    def reset(self):
        self.direction = 'RIGHT'
        self.position = [WIDTH // 2, HEIGHT // 2]
        self.body = [list(self.position)]
        self.score = 0
        self.frame_iteration = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randrange(0, WIDTH // self.size) * self.size
        y = random.randrange(0, HEIGHT // self.size) * self.size
        self.food = [x, y]
        if self.food in self.body:
            self._place_food()

    def check_collision_at(self, pt):
        if pt[0] < 0 or pt[0] > WIDTH - self.size or pt[1] < 0 or pt[1] > HEIGHT - self.size:
            return True
        if pt in self.body[1:]:
            return True
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0
        done = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.body.insert(0, list(self.position))

        if self.check_collision_at(self.position) or self.frame_iteration > 100 * len(self.body):
            done = True
            reward = -10
            return reward, done, self.score

        if self.position == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.body.pop()

        screen.fill(COLORS['black'])
        for pos in self.body:
            pygame.draw.rect(screen, COLORS['gold'], pygame.Rect(pos[0], pos[1], self.size, self.size))
        pygame.draw.rect(screen, COLORS['red'], pygame.Rect(self.food[0], self.food[1], self.size, self.size))
        self._draw_score()
        pygame.display.flip()
        clock.tick(self.speed)

        return reward, done, self.score

    def _move(self, action):
        """
        action: [1,0,0] -> skręt w lewo
                [0,1,0] -> prosto
                [0,0,1] -> skręt w prawo
        """
        directions = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            idx = (idx - 1) % 4
        elif np.array_equal(action, [0, 1, 0]):
            idx = idx 
        else:
            idx = (idx + 1) % 4

        self.direction = directions[idx]

        x, y = self.position
        if self.direction == 'RIGHT':
            x += self.size
        elif self.direction == 'LEFT':
            x -= self.size
        elif self.direction == 'UP':
            y -= self.size
        elif self.direction == 'DOWN':
            y += self.size

        self.position = [x, y]

    def _draw_score(self):
        font = pygame.font.SysFont('times new roman', 20)
        score_surface = font.render('Score : ' + str(self.score), True, COLORS['white'])
        score_rect = score_surface.get_rect()
        score_rect.midtop = (WIDTH / 10, 15)
        screen.blit(score_surface, score_rect)
