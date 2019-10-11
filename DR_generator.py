import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn
import datetime

# Synthetic data generator

class hermetic_generator:
    '''
    This class is used to generate synthetic dataset of typical and atypical curves (containing hermetic regions).

    This generator assumes hermetic curves start high (cell viab probabilty = 1) then follow a logisitic shape during the inhibitory period followed a refractory/hermetic region wherin the curve shape is assumed to be linearly increasing.
          ___
      p  |    \
      r  |     \         _/
      o  |      \      _/
      b  |       \____/
         |___________________
                dose

    The model parameters of inhibitory and hermetic regions are randomly sampled from all aproporiate values.

    # Good inuition for logistic regression parameter choices: http://mathworld.wolfram.com/LogisticEquation.html

    Inhibitory region model: Logisitic/Sigmoid

    P_i(c) =   1/(1+exp(-(b0 + b1*c)))
        where,
            c = concentration
            b0 = intercept param.
            b1 = slope param.

    Hermetic region model: linear increasing

    P_h(c) = m*c + b     for p_h(c) < 1
           = 1           otherwise
        where,
            m = slope > 0
            c = concentration
            b = y intercept

    We expect there to be both biological and measurement variation which we will model as a Gaussian distribution; The overall piecewise probability function is:

    P(c) = N(u, s)
        where,
            s = sampling std (user defined)
            u = distribution mean, centered at:
                P_i(c) for 0 <= c <= t
                P_h(c) for t <= c < max_dose
                    where,
                        t = transition point

    we also expect our equation to be discrete continuous (we are pulling only 7 points at discrete concentration points from these models) such that:

        P_i(t) = P_h(t)

    '''

    def __init__(self, s=0.05, b1range = (-8., 0.), mrange = (0., 0.5), b0range = (-10,0), trange=(0.001,10), dose_points = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]):
            '''

            inputs
                s <float> standard deviation for gaussian noise model
                rrange <tuple, floats> should be negative, the allowed range of values to choose for r in the inhibitory logistic equation
                dose_points <list> the concentration points to sample at. First value should always be zero.
                mrange <tuple, floats> must be positive, allowed range of values to choose for m in the hermetic model
                b0range <tuple, float> range of acceptable b0's to sample from
                trange <tuple, float> range of transition point to sample from, neither value can be zero (sampled in logspace)

            outputs
                object handle
            '''

            #assert dose_points[0] == 0, 'dose_points should be a list whose first value is 0.0'

            self.s = s
            self.b1range = b1range
            self.mrange = mrange
            self.b0range = b0range
            self.trange = trange
            self.dose_points = np.array(dose_points)

    def __generate(self, t, b1, b0, m, n=1, plot=True):
        '''
        generates a single set of dose response data.

        inputs
            t <float> inhibitory -> hermetic transition point
            b <float> negative value defining logistic equation shape
            b0 <float> [0,1] expected to be close to 1. Naive = 1. Value defining probability at no dose.
            m <float> linear model slope, should be positive [increasing]
            n <int> number of curves to generate from these parameters - only variation will be noise - [placeholder currently]

        outputs
             dr <pandas dataframe> columns with names ['t', 'b1', 'b0', 'm', 'b', *self.dose_points]
                                                                                        |
                                                                  _____________________/ \____________________
                                                                 / cell viablity values [0,1] over dose points \
        '''

        # solve for b such that piecewise model is equal at t
        b = 1 / (1 + np.exp(-(b0 + b1* np.log10(t)))) - m*np.log10(t)

        # we fit our models in logspace
        c = self.dose_points

        # inhibitory means
        #ui = 1 / (1 + ((1/c0) - 1)*np.exp(-r*c))
        ui = 1 / (1 + np.exp(-(b0 + b1* np.log10(c))))

        # hermetic means
        uh = np.array( [min(h, 1) for h in m*np.log10(c) + b] )
        #b02 = -b0 - b1*np.log10(c) + (m+2)*np.log10(c)
        #uh = 1 / (1 + np.exp(-(b02 + (m+2)* np.log10(c))))

        dr = np.array([[np.random.normal(loc = i, scale=self.s) if c <= t else np.random.normal(loc = h, scale=self.s) for c,i,h in zip(self.dose_points, ui, uh)] for i in range(n)])

        if plot:
            x = np.logspace(-3,np.log10(max(self.dose_points)), 100)
            pi = 1 / (1 + np.exp(-(b0 + b1*np.log10(x))))
            ph = np.array( [min(h, 1) for h in m*np.log10(x) + b])
            #ph = 1 / (1 + np.exp(-(b02 + (m+2)* np.log10(x))))
            dat = pd.DataFrame({'pi':pi, 'ph':ph, 'x':x})
            f, ax = plt.subplots(1,1,figsize=(10,10))
            sbn.lineplot(x='x', y='pi', label='inhib. model', data=dat, ax=ax)
            sbn.lineplot(x='x', y='ph', label='herm. model', data=dat, ax=ax)
            [ax.plot(self.dose_points, d, 'go', alpha= 0.5, label='sampled pts') if i==0 else ax.plot(self.dose_points, d, 'go', alpha= 0.5,) for i,d in enumerate(dr)]
            ax.set_ylim([-0.5,1.5])
            ax.axvline(x=t, color='red', label='transition point')
            ax.set_xscale('log')
            ax.set_title('t [%.2f], b1 [%.2f], b0 [%.2f], m [%.2f], b [%.2f]' %(t,b1,b0, m, b))
            plt.legend()
            plt.show()

        return pd.DataFrame({**{'t':t, 'b1':b1, 'b0':b0, 'm':m,'b':b, 's':self.s},**{'DOSE_%.2f'%d:p for d,p in zip(self.dose_points, dr.T)}})


    def get(self, n, nn=1, plot=False):
        '''
        function to build synthetic dataset.

        inputs
            n <int> number of independant dose-response models to sample from
            nn <int> number of observations from each model to return

        outputs
             dr <pandas dataframe> columns with names ['id', t', 'b1', 'b0', 'm', 'b', *self.dose_points]
                 [shape: (n*nn, 5 + len(self.dose_points))]                             |
                 [id <str>: seed_time_nn]                         _____________________/ \____________________
                                                                 / cell viablity values [0,1] over dose points \

        UH OH... hitting a few issues:

            Need to add a min inhibitory value, I suspec that'll be important in defining curve shape.
            [DONE] THIRD: need to make sure Ph(c=max_dose) <= 1, can't have huge numbers...Switch to using positive sigmoid for hermetic model? Kept it linear, used max of 1.
            [DONE] FOURTH: doses are log10 distributed so transition point t should maybe be sampled with log weights as well! or ... should it? kind of explains the proportion of tail occurence...
            [DONE] FIFTH: My logistic and linear models usually are fit in the log space, so I should really be doing the same here, right? look into that.
        '''

        first = True
        dat = None
        for i in range(n):
            if (n%1000==0):
                print('progress: %.2f%%' %(i/n*100), end='\r')
            state = np.random.get_state()
            t = np.power(10, np.random.uniform(low = np.log10(min(self.trange)), high=np.log10(max(self.trange)))) # sample uniformly in logspace between first nonzero dose point
            b1 = np.random.uniform(low = min(self.b1range), high=max(self.b1range))
            b0 = np.random.uniform(low = min(self.b0range), high=max(self.b0range))
            m = np.random.uniform(low = min(self.mrange), high = max(self.mrange))
            obs = self.__generate(t,b1,b0,m,n=nn, plot=plot).assign(id='%s_%s_%d' %(str(state[0]), str(datetime.datetime.now().time()), nn))
            dat = obs if first else dat.append(obs, ignore_index=True)
            if first: first = False

        return dat
