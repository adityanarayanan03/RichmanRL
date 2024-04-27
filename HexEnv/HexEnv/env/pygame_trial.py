import pygame
import math

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 11
SCREEN_HEIGHT = BOARD_SIZE*50
SCREEN_WIDTH = SCREEN_HEIGHT*1.5
HEX_RADIUS = SCREEN_HEIGHT / BOARD_SIZE / 2  # Radius of the circumscribed circle

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Hex Board")

# Function to draw a circle at given coordinates
def draw_circle(x, y, color):
    pygame.draw.circle(screen, color, (x, y), HEX_RADIUS)

# Function to draw the hex board
def draw_board(rows, cols):
    for row in range(rows):
        for col in range(cols):
            offset_x = 2*HEX_RADIUS * col + HEX_RADIUS*row + HEX_RADIUS
            offset_y = HEX_RADIUS + row*HEX_RADIUS*math.sqrt(3)
            draw_circle(offset_x, offset_y, WHITE)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill(BLACK)  # Fill the screen with black color
    draw_board(BOARD_SIZE, BOARD_SIZE)  # Draw an 8x8 hex board
    pygame.display.flip()  # Update the display

# Quit Pygame
pygame.quit()
