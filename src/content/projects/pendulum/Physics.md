---
title: "Lagrangian"
description: "dawdw"
pubDate: 2026-05-01
---

## Variables

- $\theta$ = pendulum angle
- $\phi$ = wheel angle with respect to pendulum
- $I_p$ = moment of inertia of pendulum
- $I_w$ = moment of inertia of wheel
- $b_{w}$ = friction torque of wheel
- $b_{p}$ = friction torque of pendulum
- $m$ = mass of system
- $g$ = gravity
- $l$ = length to center of mass

To use the Lagrange-Euler Equation we have to define the Lagrangian.
The kinetic energy of the system is:
$$
T = \frac{1}{2}I_{w}(\dot{\theta}+ \dot{\phi})^2+ \frac{1}{2}I_{p}(\dot{\theta})
$$
This is the relative velocity of the wheel attached to the pendulum $\dot{\theta}+ \dot{\phi}$.
Then we define the potential function:
$$
V = mgl\cos(\theta)
$$
Where $m$ is the mass of the system, and $l$ is the distance from the pivot to the center of mass.
The Lagrangian equation is:
$$
\mathcal{L} = T - V
$$
$$
\mathcal{L} = \frac{1}{2}I_{w}(\dot{\theta}+ \dot{\phi})^2+ \frac{1}{2}I_{p}(\dot{\theta}) - mgl\cos(\theta)
$$

## Lagrange-Euler Equation

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} - \frac{\partial \mathcal{L}}{\partial q} = Q_{i}^{NC}$$
We define $q = \begin{bmatrix} \theta \\ \phi \\ \end{bmatrix}$
Expanding the $Q_{i}^{NC}$ term, we have two contributions, the torque of the brushless motor and the friction on the wheel and the body pendulum.

So expanding the Lagrange-Euler Equation becomes:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} - \frac{\partial \mathcal{L}}{\partial q} = \tau_{i} + Q_{i}^{diss}$$
Using the Rayleigh Dissipation Function:
$$
\mathcal{R} =\frac{1}{2}b_{p}\dot{\theta}^2 + \frac{1}{2}b_w \dot{\phi}^2
$$
Where $b_{p}$ and $b_{w}$ are coefficients of friction respectively of pendulum on the pole and wheel.
Then 
$$
Q_{i}^{diss} = - \frac{\partial \mathcal{R}}{\partial{\dot{q}}}
$$
### Brushless motor
![[Screenshot 2026-05-07 at 14.56.55.png]]
We can model our brushless motor as a normal DC motor with a good approximation, drawing a typical circuit we can easily identify the torque function of the input voltage.

$$
\tau = \frac{ K_t V_{a}}{R_{a}} - \frac{ K_t K_e \dot{\phi}}{R_{a}}

$$

### Derivation
component $\theta$:
$$
\frac{\partial \mathcal{L}}{\partial \dot{\theta}} = I_{w}(\dot{\theta}+\dot{\phi})+I_{p}(\dot{\theta})=\dot{\theta}(I_{p}+I_{w})+I_{w}\dot{\phi}
$$
$$
\frac{\partial}{\partial t} \frac{\partial \mathcal{L}}{\partial \dot{\theta}} = \ddot{\theta}(I_{p}+I_{w})+ I_{w}\ddot{\phi}
$$
$$
\frac{\partial \mathcal{L}}{\partial \theta} = mgl\sin(\theta)

$$
The result for the $\theta$ component is:
$$
\ddot{\theta}(I_{p}+I_{w})+ I_{w}\ddot{\phi} - mgl\sin(\theta) = \tau_{\theta} + Q_{\theta}^{diss}
$$
Where:
$$
Q_{\theta}^{diss} = - \frac{\partial \mathcal{R}}{\partial{\dot{\theta}}}=-b_{p}\dot{\theta}
$$
The motor applies $+\tau$ to the wheel and an opposite $-\tau$ to the pendulum.
Since the angle $\phi$ is relative to the position of the pendulum, the work depends on relative displacement.
This is the reason why the internal motor does not net work on global rotation.
For this reason:
$$
\tau_{\theta} = 0
$$
The final kinematic equation for the $\theta$ component is:
$$
\ddot{\theta}(I_{p}+I_{w})+ I_{w}\ddot{\phi} - mgl\sin(\theta) = -b_{p}\dot{\theta}

$$
Component $\phi$:
$$
\frac{\partial \mathcal{L}}{\partial \dot{\phi}} = I_{w} \dot{\phi}+I_{w}\dot{\theta}
$$
$$
\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{\phi}} = I_{w} \ddot{\phi}+I_{w}\ddot{\theta}
$$
Inserting the torque $\tau_{\phi}$ and the friction from the Reiligh dissipation function:
$$
I_{w} \ddot{\phi}+I_{w}\ddot{\theta} = \frac{ K_t V_{a}}{R_{a}} - \frac{ K_t K_e \dot{\phi}}{R_{a}} - b_{w}\dot{\phi}
$$
Now that we have our system of kinematics equation:

$$
\begin{cases}
  \ddot{\theta}(I_{p}+I_{w})+ I_{w}\ddot{\phi} - mgl\sin(\theta) = -b_{p}\dot{\theta} \\
  I_{w} \ddot{\phi}+I_{w}\ddot{\theta} = \frac{ K_t V_{a}}{R_{a}} - \frac{ K_t K_e \dot{\phi}}{R_{a}} - b_{w}\dot{\phi}
\end{cases}

$$
This dynamic system is not linear, we can linearise it near an equilibrium point, that is when the pendulum is straight up ($\theta = 0$), so that when the angle is small enough we can solve an **optimal control** problem like LQR.
For small angles $\sin\theta \approx \theta$ so our linearised system when the pendulum is straight up becomes:
$$

\begin{cases}
  \ddot{\theta}(I_{p}+I_{w})+ I_{w}\ddot{\phi} - mgl\theta = -b_{p}\dot{\theta} \\
  I_{w} \ddot{\phi}+I_{w}\ddot{\theta} = \frac{ K_t V_{a}}{R_{a}} - \frac{ K_t K_e }{R_{a}}\dot{\phi} - b_{w}\dot{\phi}
\end{cases}

$$
The equation of motions must be represented in matrix form:

$$
\dot x(t) = Ax(t)+Bu(t)
$$
Defining the state vector with our angles and angular velocities and input, that is the voltage of the motor:
$$
x = \begin{bmatrix} \theta & \dot{\theta} & \phi & \dot{\phi}\end{bmatrix}^T
$$
where $x_1 = \theta, \quad x_2 = \dot{\theta}, \quad x_3 = \phi, \quad x_4 = \dot{\phi}$ and:
$$
V_{a} = u
$$
Defining the scalars:
$$
E = \frac{K_{t}}{R_{a}}
$$
$$
D = \frac{K_{t}K_{e}}{R_{a}}+ b_{w}
$$
Substituting into the system:
$$
\begin{cases}
  \dot{\,\,x_{2}}(I_{p}+I_{w})+ I_{w}\dot{\,\,x_{4}} - mgl \, x_{1} = -b_{p}\,x_{2} \\
  \,\,I_{w} \dot{\,\,x_{4}}+I_{w}\dot{\,\,x_{2}} = E\,u - D \, x_{4} - b_{w}x_{4}
\\ \dot{\,\,x_{1}} = x_{2} \\
\dot{\,\,x_{3}} = x_{4}
\end{cases}
$$
Solving for $\dot{\,\,x_{2}}$ in the second equation we get:

$$
\dot{\,\,x_{2}} = \frac{E}{I_{w}}u-\frac{D}{I_{w}}x_{4}-\dot{\,\,x_{4}}
$$
And substituting this in the first one we get:
$$
\dot{\,\,x_{4}} = -\frac{mgl}{I_{p}}x_{1} + \frac{b_{p}}{I^p}\,x_{2} - \frac{D(I_{w} + I_{p})}{I_{w}\,I_{p}}\, x_{4}+\frac{E\,(I_{w}+I_{p})}{I_{w}\,I_{p}} u 
$$
And substituting back this into the $\dot{\,\,x_{2}}$ after some algebra we get:
$$
\dot{\,\,x_{2}} = \frac{mgl}{I_{p}}x_{1} - \frac{b_{p}}{I^p}\,x_{2} + \frac{D}{\,I_{p}}\, x_{4}-\frac{E\,}{\,I_{p}} u 
$$
Now we can write the dynamic matrix A around the equilibrium point:

$$

A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
\frac{mgl}{I_{p}} & -\frac{b_{p}}{I^p} & 0 & \frac{D}{\,I_{p}} \\
0 & 0 & 0 & 1 \\
-\frac{mgl}{I_{p}} & \frac{b_{p}}{I^p} & 0 & - \frac{D(I_{w} + I_{p})}{I_{w}\,I_{p}}
\end{bmatrix}

$$
And the B matrix:

$$
B = \begin{bmatrix}
0 \\
-\frac{E\,}{\,I_{p}} \\
0 \\
\frac{E\,(I_{w}+I_{p})}{I_{w}\,I_{p}}
\end{bmatrix}

$$

