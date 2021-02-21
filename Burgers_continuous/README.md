# Burgers continuous

This folder contains a solver focusing on the following PDE (viscous Burgers' equation):

$$
\begin{cases}
u+uu_t-(0.01/\pi)u_{xx}=0,\\
u(x,0)=-\sin(\pi x),\\
u(-1,t)=u(1,t)=0,
\end{cases}
$$

whose analytical solution (on discrete points) is available in `burgers_shock.mat`.

Run `Burgers_continuous.py`, which will produce a txt file `tmp.txt` containing the approximated solution. Then run `plot.py` to visualize the solution, which has been saved in `Figure_1.png` in advance.

Note: all the files not mentioned above are only for testing or debugging.

## Requirements

* python==3.8.5
* pytorch==1.7.1
* numpy==1.19.2
* scipy==1.5.2
* matplotlib==3.3.2 (only for plotting)
