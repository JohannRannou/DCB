"""
Variation de delta imposé
-> conséquence sur G
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.gridspec as gridspec

E = 130e9
h = 2.e-3
b = 20.e-3
I = b*h**3/12.
two_gamma = 100.
Gc=two_gamma


def k(a):
    return 3/2*E*I/a**3

def deflection(x,delta,a):

    P = k(a)*delta
    a = x[-1]
    print('LOL ',x) 
    print('LOL ',a,P) 
    y = P/(E*I)*(x**3/6-a**2/2*x+1/3*a**3)
    return y
     
def Welast(a, delta):
    return 3/4*E*I/a**3*delta**2

def G_dep_impose(a, delta):
    return 9/4*E*I/b/a**4*delta**2

def G_dep_impose_reverse(G, delta):
    return ((9/4*E*I/b*delta**2)/G)**(1/4)
    
def animation():
    # fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25)
    ax1 = plt.subplot(gs[0, 0]) # row 0, col 0
    
    ax2 = plt.subplot(gs[0, 1]) # row 0, col 1
    
    ax3 = plt.subplot(gs[1, :]) # row 1, span all columns
    


    delta0 = 0.2e-3
    #Longueur de fissure initiale
    global a_max, delta_max
    a_max = G_dep_impose_reverse(Gc,delta0)
    delta_max = delta0
    a = np.linspace(2e-3,100e-3, 10000)


    ### Plot G
    delta_plot, = ax1.plot(a*1000,G_dep_impose(a,delta0), label='delta_imp={:.1f}mm'.format(delta0*1000))
    Gc_plot, = ax1.plot(a*1000,np.ones_like(a)*Gc, label=r'$G_c$',color='k', lw=2)
    a_point_plot, = ax1.plot([G_dep_impose_reverse(Gc,delta0)*1000],Gc,'o',ms=8,color='r')
    a2_point_plot, = ax1.plot([G_dep_impose_reverse(Gc,delta0)*1000],Gc,'o',ms=12,color='#00ccff', zorder=-1)

    a_line_plot, = ax1.plot([G_dep_impose_reverse(Gc,delta0)*1000,G_dep_impose_reverse(Gc,delta0)*1000],
                            [Gc,0],'--',color='r')
    ax1.set_xlabel(r'$a$ en mm')
    ax1.set_ylabel(r'$G$ en $J/m^2$')

    delta_annot = ax1.annotate(r'$\delta$={}mm'.format(delta0*1000), (G_dep_impose_reverse(300.,delta0)*1000,300),fontsize=12.)
    Gc_annot = ax1.annotate(r"$G_c$={}$J/m^2$".format(Gc), (2e-3*1e3,Gc+10),fontsize=12.)
    a_annot = ax1.annotate(r"$a$={0:2.1f}mm".format(G_dep_impose_reverse(Gc,delta0)*1000), (G_dep_impose_reverse(Gc,delta0)*1000,0.),fontsize=12.)
    
    for delta in [0.1e-3, 0.5e-3, 1e-3, 2e-3, 3e-3, 4e-3]:
        ax1.plot(a*1000,G_dep_impose(a,delta), '--', label='delta_imp={:.1f}mm'.format(delta*1000),color='grey',lw=1)    
        # ax1.annotate(r"$\delta$={0:2.2f}mm".format(delta*1000), (G_dep_impose_reverse(400.,delta)*1000,400.),fontsize=12.)

    ax1.set_ylim([0,500])


    ### F-d
    delta_space = np.linspace(0., 10e-3, 1000)
    delta_space_elast = np.linspace(0.,delta0 )
    def P(d,Gc_=Gc):
        P = (3/2)**(-1/2)*b**(3/4)*(E*I)**(1/4)*Gc_**(3/4)*d**(-1/2)
        return P

    def P_min(d,a,Gc_=Gc):
        P = (3/2)**(-1/2)*b**(3/4)*(E*I)**(1/4)*Gc_**(3/4)*d**(-1/2)
        return min(P,k(a)*d)
    
    F_d_propa_plot, = ax2.plot(delta_space*1e3,P(delta_space), label='propa',color='k')
    F_d_elast_plot, = ax2.plot(delta_space_elast*1e3,delta_space_elast*k(a_max), label='propa',color='k')
    F_d_solution_plot, = ax2.plot([delta0*1000],[P(delta0)],'o',ms=10,color='#00ccff')
    ax2.set_xlabel(r'$\delta$ en mm')
    ax2.set_ylabel(r'$P$ en N')
    ax2.set_ylim([0,500])
    ax2.set_ylim([0,150])

    
    ### Plot deflection
    x_range = np.linspace(0., a_max)
    top_deflection_plot, = ax3.plot(x_range*1000,deflection(x_range,delta0,a_max)*1000, label=r'deflection',color='k', lw=5)
    bottom_deflection_plot, = ax3.plot(x_range*1000,-1.*deflection(x_range,delta0,a_max)*1000, label=r'deflection',color='k', lw=5)
    ax3.set_xlim([0,100])
    ax3.set_ylim([-2.5,2.5])
    ax3.set_xlabel(r'$x$ en mm')
    ax3.set_ylabel(r'deflection en mm')

    ### Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_delta = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_Gc = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    slider_delta = Slider(ax_delta, 'delta (loading)', 0., 5e-3, valinit=delta0)
    slider_Gc = Slider(ax_Gc, 'Gc (moving Gc implies reset a)', 10, 500, valinit=Gc)

    ### Reset Button

    ax_reset = plt.axes([0.75, 0.025, 0.15, 0.05])
    #Je ne comprends vraiment pas pourquoi j'ai besoin de le mettre global
    global button_reset
    button_reset = Button(ax_reset, 'Close crack', color=axcolor, hovercolor='0.5')

    ### Sliders management
    def update(val):
        delta = slider_delta.val
        Gc = slider_Gc.val
        global a_max, delta_max
        delta_max = max(delta,delta_max)
        a_max = min(G_dep_impose_reverse(Gc,delta)*1000,a_max)
        #Graphe G

        delta_plot.set_ydata(G_dep_impose(a,delta))
        Gc_plot.set_ydata(np.ones_like(a)*Gc)
        a_max = max(G_dep_impose_reverse(Gc,delta), a_max)
        a_point_plot.set_xdata([a_max*1000])
        a_point_plot.set_ydata([Gc])
        a2_point_plot.set_xdata([a_max*1000])
        a2_point_plot.set_ydata([G_dep_impose(a_max, delta)])
        a_line_plot.set_xdata([a_max*1000,a_max*1000])
        a_line_plot.set_ydata([Gc,0])


        delta_annot.set_text(r'$\delta$={0:2.2f}mm'.format(delta*1000))
        delta_annot.set_position((G_dep_impose_reverse(300.,delta)*1000,300.))
        a_annot.set_text("a={0:2.1f}mm".format(a_max*1000))
        a_annot.set_position((a_max*1000,0.))

        # Gc_annot.set_text(r"$G_c$={0:2.0f}$J/m^2$".format(Gc))
        Gc_annot.set_text("G_c={0:2.0f} J/m^2".format(Gc))
        Gc_annot.set_position((2e-3*1e3,Gc+10))


        #Graph F-d
        F_d_propa_plot.set_ydata(P(delta_space,Gc))
        delta_space_elast = np.linspace(0.,delta_max )
        F_d_elast_plot.set_xdata(delta_space_elast*1000)
        F_d_elast_plot.set_ydata(delta_space_elast*k(a_max))
        F_d_solution_plot.set_xdata([delta*1000])
        F_d_solution_plot.set_ydata([min(P(delta,Gc),delta*k(a_max))])
        
        fig.canvas.draw_idle()

        #Graph deflection
        x_range = np.linspace(0., a_max)
        top_deflection_plot.set_xdata(x_range*1000)
        top_deflection_plot.set_ydata(deflection(x_range,delta,a_max)*1000)
        bottom_deflection_plot.set_xdata(x_range*1000)
        bottom_deflection_plot.set_ydata(-1.*deflection(x_range,delta,a_max)*1000)

    def update2(val):
        Gc = slider_Gc.val
        global a_max
        a_max = G_dep_impose_reverse(Gc,delta0)
        slider_delta.reset()
        update(val)
        
    slider_delta.on_changed(update)
    slider_Gc.on_changed(update2)

    ### Reset management
    def reset(event):
        global a_max
        a_max = G_dep_impose_reverse(Gc,delta0)
        slider_delta.reset()
    button_reset.on_clicked(reset)


# a = np.linspace(10e-3,100e-3, 10000)
# plt.figure()
# plt.ylim([0,500])
# for delta in [0.5e-3, 1e-3, 2e-3, 3e-3, 4e-3]:
#     plt.plot(a*1000,G_dep_impose(a,delta), label='delta_imp={:.1f}mm'.format(delta*1000))

# plt.xlabel(r'$a$ en mm')
# plt.ylabel(r'$G$ en $J/m^2$')
# plt.legend()


# plot_F_d__W_rest()


animation()

plt.show()
