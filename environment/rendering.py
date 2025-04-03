import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Only expose the Renderer class to prevent import conflicts
__all__ = ['Renderer']

class Renderer:
    def __init__(self, env):
        self.env = env
        self.window_initialized = False
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(800, 800)
        glutCreateWindow(b"Crop Monitoring Drone")
        glClearColor(1.0, 1.0, 1.0, 1.0)
        self.window_initialized = True
        glutDisplayFunc(self.display)
        
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.env.grid_size, 0, self.env.grid_size)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Draw crops
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                health = self.env.health_map[i,j]
                if health < 0.3: 
                    glColor3f(0.8, 0.2, 0.2)
                elif health < 0.6: 
                    glColor3f(0.8, 0.8, 0.2)
                else: 
                    glColor3f(0.2, 0.8, 0.2)
                
                glBegin(GL_QUADS)
                glVertex2f(i, j)
                glVertex2f(i+1, j)
                glVertex2f(i+1, j+1)
                glVertex2f(i, j+1)
                glEnd()
        
        # Draw drone
        x, y = self.env.drone_pos
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(x+0.5, y+0.9)
        glVertex2f(x+0.1, y+0.1)
        glVertex2f(x+0.9, y+0.1)
        glEnd()
        
        glColor3f(1, 1, 1)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x+0.5, y+0.9)
        glVertex2f(x+0.1, y+0.1)
        glVertex2f(x+0.9, y+0.1)
        glEnd()
        
        glutSwapBuffers()
    
    def render(self):
        if not self.window_initialized:
            self.initialize_window()
        glutPostRedisplay()
        glutMainLoopEvent()

    def initialize_window(self):
        """Set up the GLUT window and OpenGL context"""
        if not self.window_initialized:
            # GLUT initialization
            glutInit()
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(800, 800)
            glutInitWindowPosition(
                (glutGet(GLUT_SCREEN_WIDTH) - 800) // 2,
                (glutGet(GLUT_SCREEN_HEIGHT) - 800) // 2
            )
            
            # Window creation
            self.window_id = glutCreateWindow(b"Crop Monitoring Drone")
            print(f"OpenGL window created with ID: {self.window_id}")
            
            # Setup callbacks
            glutDisplayFunc(self.display)
            glutIdleFunc(self.display)
            
            # Basic OpenGL setup
            glClearColor(1.0, 1.0, 1.0, 1.0) 
            glEnable(GL_DEPTH_TEST)
            
            self.window_initialized = True
            glutMainLoop()


    def close(self):
        """Clean up resources"""
        if self.window_id is not None:
            glutDestroyWindow(self.window_id)
            self.window_initialized = False
            print("OpenGL window closed")