import matplotlib.pyplot as plt

class ABCplot(object):
    def __init__(self,abcclass):
        self.abc=abcclass
        self.bins = 30

    def hist(self):
        fig=plt.figure(figsize=(10,5))
        ax=fig.add_subplot(211)
        ax.hist(self.abc.x,bins=self.bins,label="$\epsilon$="+str(self.abc.epsilon),density=True,alpha=0.5)
        ax.hist(self.abc.xres(),bins=self.bins,label="resampled",density=True,alpha=0.2)
        
#        alpha=alpha0+abc.nsample
#        beta=beta0+Ysum
#        xl = np.linspace(gammafunc.ppf(0.0001, alpha,scale=1.0/beta),gammafunc.ppf(0.9999, alpha,scale=1.0/beta), 100)
#        ax.plot(xl, gammafunc.pdf(xl, alpha, scale=1.0/beta),label="analytic")
        plt.xlabel("$\lambda$")
        plt.ylabel("$p_\mathrm{ABC} (\lambda)$")
        plt.legend()

    def weight(self):
        ax=fig.add_subplot(212)
        ax.plot(self.abc.x,self.abc.w,".")
        plt.xlabel("$\lambda$")
        plt.ylabel("$weight$")


    def show(self):
        plt.show()
