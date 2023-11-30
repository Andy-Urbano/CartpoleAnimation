#Imports
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
import pygame, sys
from pygame.locals import *
import initialization

#Initialzing 
import os
os.environ['SDL_VIDEO_ACCELERATION'] = '0'

pygame.init()

#Setting up FPS 
FPS = 1000  

#Creating colors
RED   = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

#Other Variables for use in the program
SCREEN_HEIGHT = 1500
SCREEN_WIDTH = int(1.5*SCREEN_HEIGHT)

unit = SCREEN_HEIGHT//20

# Create the screen surface
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

#Setting up Fonts
font = pygame.font.SysFont("Verdana", 60)
font_small = pygame.font.SysFont("Verdana", 20)

#Create a white screen 
DISPLAYSURF = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("Cartpole Simulation")

class Display(pygame.sprite.Group):
    class VerticalDivision(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.image = pygame.Surface((10, SCREEN_HEIGHT//2))
            self.image.fill(BLACK)
            self.rect = self.image.get_rect(top = 0, left = 2*SCREEN_WIDTH//3)
        
        def update(self):
            pass

    class HorizontalDivision(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.image = pygame.Surface((SCREEN_WIDTH, 10))
            self.image.fill(BLACK)
            self.rect = self.image.get_rect(top = SCREEN_HEIGHT//2 -10, left = 0)
        
        def update(self):
            pass

    class MainPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.moving_part = None  # The moving part will be added later
            self.image = pygame.Surface((100, 100))
            self.rect = self.image.get_rect()
            self.zero_pos = (SCREEN_WIDTH//3, SCREEN_HEIGHT // 2 - 100)  # Position the main part in the middle of the screen
            self.rect.center = self.zero_pos
            self.x = 0
            self.x_speed = 0

        def update(self):
            self.rect.center = (int(unit*self.x) + self.zero_pos[0], self.zero_pos[1])

    class MovingPart(pygame.sprite.Sprite):
        def __init__(self, main_part, group=None):
            super().__init__(group)
            self.main_part = main_part
            self.image = pygame.Surface((50, 50))  # Adjust the size as needed
            self.image.set_colorkey(BLACK)  # Makes black colors transparent
            self.rect = self.image.get_rect()
            self.rect.center = self.main_part.rect.center
            self.angle = 0
            self.angle_speed = 0

        def update(self):
            # Update the position of the sprite
            center_x, center_y = self.main_part.rect.center
            radius = 300  # Change this value to change the radius of the circle
            self.rect.centerx = center_x + radius * np.cos(self.angle)
            self.rect.centery = center_y + radius * np.sin(self.angle)

            # Clear the image
            self.image.fill(BLACK)

            # Draw the circle on the image
            pygame.draw.circle(self.image, RED, (self.rect.width // 2, self.rect.height // 2), self.rect.width // 2)

    class ActionPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.image = pygame.Surface((10, SCREEN_HEIGHT//4))  # Adjust the size as needed
            self.rect = self.image.get_rect(top = SCREEN_HEIGHT//4, left = 5*SCREEN_WIDTH//6)
            #self.rect.center = (SCREEN_WIDTH // 2, 1000)  # Position the target part in the middle of the screencontroller.
            self.u = 0

        def update(self):
            self.u = np.clip(self.u, -5, 5)
            self.rect.x = int(unit*self.u) + 5*SCREEN_WIDTH//6

    class DummyPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.image = pygame.Surface((10, SCREEN_HEIGHT//4))  # Adjust the size as needed
            self.image.fill(RED)            # Fill the surface with white color
            self.rect = self.image.get_rect(top = SCREEN_HEIGHT//4, left = 5*SCREEN_WIDTH//6)
            #self.rect.center = (SCREEN_WIDTH // 2, 1000)  # Position the target part in the middle of the screen

        def update(self):
            pass
    
    class TargetPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.image = pygame.Surface((10, SCREEN_HEIGHT//4))  # Adjust the size as needed
            self.rect = self.image.get_rect(top = 0, left = SCREEN_WIDTH//3)
            #self.rect.center = (SCREEN_WIDTH//3, SCREEN_HEIGHT//4)  # Position the target part in the middle of the screen

        def update(self):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.rect.move_ip(-5, 0)
            if keys[pygame.K_RIGHT]:
                self.rect.move_ip(5, 0)

    class TextPart(pygame.sprite.Sprite):
        def __init__(self, group=None):
            super().__init__(group)
            self.font = pygame.font.Font(None, 100)  # Specify your font size
            self.image = self.font.render('Time: ' + str(0), True, BLACK)
            self.rect = self.image.get_rect(top=20, left=2*SCREEN_WIDTH//3+20)
            self.time = 0

        def update(self):
            self.image = self.font.render('Time: ' + str(np.round(self.time, 2)) + 's', True, BLACK)
            self.rect = self.image.get_rect(top=20, left=2*SCREEN_WIDTH//3+20)

    class SpikePart(pygame.sprite.Sprite):
        def __init__(self, index, group=None):
            super().__init__(group)
            spike_size = SCREEN_HEIGHT//200
            self.image = pygame.Surface((spike_size, spike_size))
            self.rect = self.image.get_rect(top = SCREEN_HEIGHT//2 +index*spike_size , left = 0)
            self.speed = 10

        def update(self):
            self.rect.move_ip(self.speed, 0)

    def __init__(self):
        super().__init__()

        self.main_part = self.MainPart(self)  # Create the main part without the moving part
        self.moving_part = self.MovingPart(self.main_part, self)  # Create the moving part with the main part
        self.main_part.moving_part = self.moving_part  # Update the main part with the moving part

        self.target_part = self.TargetPart(self)
        self.action_part = self.ActionPart(self)
        self.dummy_part = self.DummyPart(self)
        self.text_part = self.TextPart(self)

        self.add(self.VerticalDivision(self))  # Add the divisions to the group
        self.add(self.HorizontalDivision(self))

        self.spikes = []

    def update(self, state, u, spike_index_plural, time):
        x_next, x_speed_next, angle_next, angle_speed_next = state

        self.main_part.x = x_next
        self.main_part.x_speed = x_speed_next
        self.moving_part.angle = -(angle_next-np.pi/2)
        self.moving_part.angle_speed = angle_speed_next

        self.action_part.u = u

        self.text_part.time = time

        if len(spike_index_plural):
            for i in range(len(spike_index_plural)): 
                self.spikes.append(self.SpikePart(spike_index_plural[i], self))
        
        super().update()  # Call the update methods of all the sprites in the group
    
    def draw(self, surface):
        # Draw all the sprites in the group
        super().draw(surface)

        # Draw a line connecting the centers of the main part and the moving part
        pygame.draw.line(surface, BLACK, self.main_part.rect.center, self.moving_part.rect.center, 5)


class Cartpole():
    def __init__(self):
        self.M = 5
        self.m = 1
        self.L = 2
        self.g = -9.8
        self.d = 0
        self.dt = 1/FPS

        self.A, self.B = initialization.Cartpole_init(self.m, self.M, self.L, self.g, self.d)
        self.C = np.array([[1, 0, 0, 0], 
                           [0, 0, 0, 0], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 0]])
        
        self.Q = np.eye(4)
        self.Q[0, 0] = 10
        self.Q[2, 2] = 1000
        self.R = 0.01*np.eye(1)
        self.Kc = initialization.Control_init(self.A, self.B, self.Q, self.R)
        self.Kf = initialization.Kalman_init(self.A, self.C)

        self.state = np.array([0, 0, np.pi+ 0.01, 0])

    def cartpole_der(self, x, u=0):
        Sx=np.sin(x[2])
        Cx=np.cos(x[2])
    
        D = self.m*self.L*self.L*(self.M+self.m*(1-Cx**2))
        
        dx_1 = x[1]
        dx_2 = (1/D)*(-self.m**2*self.L**2*self.g*Cx*Sx + self.m*self.L**2*(self.m*self.L*x[3]**2*Sx - self.d*x[1])) + self.m*self.L*self.L*(1/D)*u
        dx_3 = x[3] 
        dx_4 = (1/D)*((self.m+self.M)*self.m*self.g*self.L*Sx - self.m*self.L*Cx*(self.m*self.L*x[3]**2*Sx - self.d*x[1])) - self.m*self.L*Cx*(1/D)*u
  
        dxdt=np.array([dx_1, dx_2, dx_3, dx_4])
        
        return dxdt
    
    def cartpole_RK4(self, x, u=0):
        k1 = self.cartpole_der(x, u)
        k2 = self.cartpole_der(x + 0.5*self.dt*k1, u)
        k3 = self.cartpole_der(x + 0.5*self.dt*k2, u)
        k4 = self.cartpole_der(x + self.dt*k3, u)
        x_next = x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return x_next
    
    def update(self, z, u):
        z_next = self.cartpole_RK4(z, u)
        self.state = z_next


class LQGController:
    def __init__(self, A, B, C, Kf, Kc, dt):
        self.A = A
        self.B = B
        self.C = C
        self.Kf = Kf
        self.Kc = Kc
        self.dt = dt

        self.Omega_mu = self.A - self.B @ self.Kc - self.Kf @ self.C
        self.Omega_z = self.B @ self.Kc

        self.M = np.zeros((8, 8))
        self.M[0:4, 0:4] = self.Omega_mu
        self.M[0:4, 4:8] = self.Omega_z

        self.m = np.zeros((8, 1))
        self.m[2] = 0.01

        self.helper = np.zeros((4, 8))
        self.helper[0:4, 0:4] = np.eye(4)
        self.helper[0:4, 4:8] = -np.eye(4)

        self.decode_u = - self.Kc @ self.helper

        self.u = 0
    
    def update(self, y, target=0):
        #Get the target state derivative
        target_state = np.array([[target], [0], [0], [0]])
        target_deriv = target_state - self.m[4:]

        #Get the contribution from the Kalman filter
        y = np.reshape(y, (4, 1))
        Ky = self.Kf@y

        #Make these into the inputs vector
        inputs = np.concatenate((Ky, target_deriv))

        # Calculate the control signal
        self.u = self.decode_u@self.m
        self.u = self.u[0, 0]

        # Calculate the derivative of the estimated state and update it
        m_dot = self.M @ self.m + inputs
        self.m += m_dot * self.dt


class SCNController:
    def __init__(self, A, B, C, Kf, Kc, dt, tau, N):
        self.A = A
        self.B = B
        self.C = C
        self.Kf = Kf
        self.Kc = Kc
        self.dt = dt
        self.tau = tau
        self.N = N

        self.Omega_mu = self.A - self.B @ self.Kc - self.Kf @ self.C
        self.Omega_z = self.B @ self.Kc

        self.M = np.zeros((8, 8))
        self.M[0:4, 0:4] = self.Omega_mu
        self.M[0:4, 4:8] = self.Omega_z

        self.D = np.random.randn(8, self.N)
        self.D = self.D / np.linalg.norm(self.D, axis=0)
        self.D = self.D / 100
        self.T = np.diag(self.D.T @ self.D) / 2

        self.V = np.zeros((self.N, 1))
        self.s = np.zeros((self.N, 1))
        self.r = np.zeros((self.N, 1))

        self.O_f = -self.D.T @ self.D
        self.O_s = self.D.T@(self.tau * np.eye(8) + self.M)@self.D

        self.m = np.zeros((8, 1))
        self.m[2] = 0.01

        self.r = np.linalg.pinv(self.D) @ self.m

        self.target_state = np.array([[0], [0], [0], [0]])

        self.helper = np.zeros((4, 8))
        self.helper[0:4, 0:4] = np.eye(4)
        self.helper[0:4, 4:8] = -np.eye(4)

        self.decode_u = -self.Kc @ self.helper

        self.u = 0

    def update(self, y, target=0):
        #Get the target state derivative
        target_state = np.array([[target], [0], [0], [0]])
        target_deriv = target_state - self.m[4:]

        #Get the contribution from the Kalman filter
        y = np.reshape(y, (4, 1))
        Ky = self.Kf@y

        #Make these into the inputs vector
        inputs = np.concatenate((Ky, target_deriv))

        # Calculate the control signal
        self.u = self.decode_u@self.D@self.r
        self.u = self.u[0, 0]

        # Calculate the next voltage
        self.V = self.V + self.dt * ( -self.tau*self.V + self.O_s@self.r + self.D.T@inputs) + self.O_f@self.s

        # Check if there are neurons whose voltage is above threshold
        above = np.where(self.V > self.T)[0]
        if len(above):
            self.s = np.zeros((self.N, 1))
            self.s[np.argmax(self.V)] = 1

        # Update rate
        self.r = (1- self.tau*self.dt) * self.r + self.s

        self.m = self.D @ self.r

#Setting up Sprites        
cartpole = Cartpole()
display = Display()
controller = SCNController(cartpole.A, cartpole.B, cartpole.C, cartpole.Kf, cartpole.Kc, cartpole.dt, 1, 100)
#controller = LQGController(cartpole.A, cartpole.B, cartpole.C, cartpole.Kf, cartpole.Kc, cartpole.dt)

spikes = []
x = []
x_mu = []
theta = []
theta_mu = []
offset = np.array([0, 0, np.pi, 0])
target = 0
counter = 0
#Game Loop
while True:
    #Get variables
    state = cartpole.state
    y = cartpole.C @ (state-offset)
    u = controller.u

    #Store variables
    x.append(state[0])
    x_mu.append(controller.m[0][0])
    theta.append(state[2])
    theta_mu.append(controller.m[2][0])

    #Update the states
    controller.update(y, target)
    cartpole.update(state, u)
    
    spikes = np.where(controller.s == 1)[0]

    counter += 1

    #Update the sprites
    if counter % 10 == 0:
        DISPLAYSURF.fill((255, 255, 255))

        #Cycles through all events occuring  
        for event in pygame.event.get(): 
            if event.type == QUIT:
                plt.plot(x_mu)
                plt.plot(x)
                plt.title('x')
                plt.show()

                plt.plot(np.array(theta_mu)+np.pi)
                plt.plot(theta)
                plt.title('theta')
                plt.show()

                pygame.quit()
                sys.exit()

        target = (display.target_part.rect.centerx - SCREEN_WIDTH//3) / unit
        display.update(state, u, spikes, counter*cartpole.dt)
        display.draw(DISPLAYSURF)
        
        pygame.display.flip()
