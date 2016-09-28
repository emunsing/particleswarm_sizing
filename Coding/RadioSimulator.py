import pandas as pd
import numpy as np

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

### USING THIS CLASS ####
""" 
- Constraints are created under initialization. Set current/voltage limits here.
- System parameters are set under createSysMatrix(). Change component parameters (resistance, capacitance, etc) here.
- Component costs are set under computeCost(). Change costs here.

Uses:
- Simulate and return a dataframe: 
       myRadioSimulator.simulate(initVariables,stopOnError=False,returnDf=True)
- Simulate and return feasibility flag: 
       myRadioSimulator.simulate(initVariables,stopOnError=False,returnDf=False)
- Simulation and calculate cost (infinite cost if infeasible): 
       myRadioSimulator.computeCost(initVariables)

Example use: 

import RadioSimulator
myRadioSimulator = RadioSimulator.RadioSimulator(radioFile = '../Data/PowerMEMS_Sample_Data_em_20160707.csv')
initVariables = {'TEGserial':5, 'TEGparallel':15, 'batts':5, 'caps':20, 'SOC':0.5, 'V_b':2, 'V_c':2}
myRadioSimulator.computeCost(initVariables)  # Compute the weighted cost

"""

class RadioSimulator:
    def __init__(self, radioFile='50step_downsampled_toy.csv'):
        
        self.df = pd.read_csv(radioFile)
        
        # Define constraints:

        V_rmin = 1.8    # Radio cutoff voltage min and max
        V_rmax = 3.6    

        I_cmax = 0.1  # Max discharge current of capacitor (positive)
        I_cmin = -0.1 # Max charge current of capacitor (should be negative)

        SOC_max= 0.8  # 
        SOC_min= 0.2  # 
        I_bmax= 0.25e-3  # Max discharge current of battery (positive)
        I_bmin= -0.25e-3 # Max charge current of battery (should be negative)

        ## Forming Constraints for roll-your-own solver/simulator
        # For each variable, define a 'min' and 'max' column which will be checked by the isFeasible() function

        self.variables = ['I_t','I_b','I_c','V','V_b','V_c','OCV','SOC','dV_b','dV_c','dSOC']

        self.constraints = pd.DataFrame(columns=self.variables)

        self.constraints.loc['min','I_t']  = 0  # Practical constraint; it's unlikely that we would ever dump power as heat
        self.constraints.loc['max','I_t']  = float('inf')

        self.constraints.loc['min','I_b']  = I_bmin
        self.constraints.loc['max','I_b']  = I_bmax

        self.constraints.loc['min', 'I_c'] = I_cmin
        self.constraints.loc['max','I_c']  = I_cmax

        self.constraints.loc['min','V']    = V_rmin
        self.constraints.loc['max','V']    = V_rmax

        self.constraints.loc['min','SOC']  = SOC_min
        self.constraints.loc['max','SOC']  = SOC_max

        #Put loose constraints on variables that are not meaningfully constrained
        self.constraints.loc['min',['V_b','V_c','OCV','dV_b','dV_c','dSOC']] = -float('inf')
        self.constraints.loc['max',['V_b','V_c','OCV','dV_b','dV_c','dSOC']] =  float('inf')
        
    def createSysMatrix(self, TEGs=2, TEGp=10, caps=1, batts=1):
        # Note: A is full rank; np.linalg.matrix_rank(A) = 11
        #dt = 1
        
        ## TEG
        # TEG Current output function: -1.7I_t - 0.032*V + 0.000136*T = 0
        # or for a given temperature difference dK,  I_t = T_0*dK - zeta * V(t)
        # where
        # This is for a device with 50 couples 
        #  T_0 = 0.000136 / 1.7
        #  zeta = 0.032 / 1.7
        # Set both parameters to 0 for testing w/o TEG
        dK = 20      # Temperature difference, K
        self.T_0 =  0.000136 / 1.7 # * dK  #TEG current intercept on I/V plot; T_0=0.0015 for dK=20
        zeta = 0.032 / 1.7  # slope of TEG current output sensitivity to voltage

        ## Capacitor
        R_c1 = 40     # Series resistance of capacitor [Ohms]
        R_c2 = 2e5    # Leakage resistance [Ohms]
        C_c  = 50e-3  # Capacitance, [Farads] or coulombs/volt

        ## Battery
        R_b1 = 200    # Battery series resistance [Ohms]
        R_b2 = 5      # Battery parallel resistance [Ohms]
        C_b  = 1e-3   # Battery internal capacitance [Farads]
        Q    = 1.8   # Battery capacity [Coulombs] - based on 0.5 mAh/cm^2 from Winslow 2013. 1mAh = 3.6C

        ## OCV model
        # Desire a roughly 1V to 3.4V range over SOC = 0.2 to SOC=1
        # (3.4-1)/(1-0.2) = 3 = slope
        # Intercept = 1; slope = 2.75V
        # OCV at 50% charge: 2.375
        self.V_0  = 1  # Intercept in linear battery model, OCV = nu * SOC + V_0
        nu   = 2.5  #  Slope of linear battery model
        self.eff =  0.9**0.5 # One-way battery efficiency
        
        #  Creating the system's matrix                                                                              .       .       .
        #               I_T(t), I_B(t), I_C(t), V(t), V_b(t), V_c(t), OCV(t), SOC(t), V_b(t), V_c(t), SOC(t)
        A = np.array( [ [TEGp,  batts , caps  ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ,  0 ],  #0  = I_R Circuit KCL  - GOOD
                        [ 1   ,  0    ,   0, zeta/TEGs, 0   ,   0   ,   0   ,   0   ,   0   ,   0   ,  0 ],  #1  = T_0 TEG - GOOD
                        [ 0   ,  0    , R_c1  ,  1  ,   0   ,  -1   ,   0   ,   0   ,   0   ,   0   ,  0 ],  #2  = 0   Cap KVL - GOOD
                        [ 0   ,  0    , 1/C_c ,  0  ,   0, 1/(C_c*R_c2),0   ,   0   ,   0   ,   1   ,  0 ],  #3  = 0   Cap KCL - GOOD
                        [ 0   ,  R_b1 ,   0   ,  1  ,   -1  ,   0   ,  -1   ,   0   ,   0   ,   0   ,  0 ],  #4  = 0   Batt KVL - GOOD
                        [ 0   , 1/C_b ,   0   ,  0 ,1/(C_b*R_b2),0  ,   0   ,   0   ,   1   ,   0   ,  0 ],  #5  = 0   Batt KCL - UNCERTAIN
                        [ 0   ,  0    ,   0   ,  0  ,   0   ,   0   ,   1   , -nu   ,   0   ,   0   ,  0 ],  #6  = V_0 OCV Linearization
                        [ 0   ,  1/Q  ,   0   ,  0  ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ,  1 ],  #7  = 0   Cons of Charge; initially assumes no efficiency loss
                        [ 0   ,  0    ,   0   ,  0  ,   0   ,   0   ,   0   ,  -1   ,   0   ,   0   ,  1 ],  #8  = -SOC(k-1)
                        [ 0   ,  0    ,   0   ,  0  ,  -1   ,   0   ,   0   ,   0   ,   1   ,   0   ,  0 ],  #9 = -V_b(k-1)
                        [ 0   ,  0    ,   0   ,  0  ,   0   ,  -1   ,   0   ,   0   ,   0   ,   1   ,  0 ] ])#10 = -V_c(k-1)
        return A
    
    def computeCost(self, initVariables=None):
        TEGcost  = 1
        capCost  = 1
        battCost = 1
        
        # feasible = self.simulate(initVariables,stopOnError=False,returnDf=False)
        (feasible,df,failStep) = self.simulate(initVariables,stopOnError=True,returnDf=True)

        if feasible:
            cost = initVariables['TEGserial']* initVariables['TEGparallel'] * TEGcost + \
                    capCost * initVariables['caps'] + battCost * initVariables['batts']
        else:
            cost = (df.shape[0]-failStep) * 100
            # cost = float('inf')
            
        return cost
        
    def simulate(self, initVariables=None,stopOnError=True,returnDf=False):
        #  expect initVariables as a dictionary with terms:
        #   variable initial states: 'SOC', 'V_b', 'V_c'
        #   design parameters: 'TEGserial', 'TEGparallel','caps', 'batts'

        """
        Simulate a scenario and return a tuple of (successful,data, failStep) based on the following information:
        A: The system matrix, capturing the dynamics of the system and created with the createSysMatrix function
        df: dataframe containing columns ['time', u'dt', u'I_r', u'SOC', u'V_b', u'V_c'] and having the 
        T_0: TEG Temp/Current output intercept 
        V_0: Battery OCV/SOC slope intercept
        variables: list of variable names used in the constraint matrix.
        ***NOTE:*** Assumes that the dataframe 'constraints' is already populated and available in the calling environment!
        ignoreConstraints: Flag to indicate whether we should continue simulating even when constraints are violated 
        """
        df = self.df.copy()
        
        df.loc[0,'SOC'] = initVariables['SOC']
        df.loc[0,'V_b'] = initVariables['V_b']
        df.loc[0,'V_c'] = initVariables['V_c']
        

        A = self.createSysMatrix(initVariables['TEGserial'], initVariables['TEGparallel'], 
                                 initVariables['caps'], initVariables['batts'])

        feasible = True
        failStep = 1

        for i in df.index[1:]:
            
            dt = df.loc[i,'dt']
            #                   I_r(t),        T_0*dK             , 0, 0, 0, 0,    V_0  , 0,    -SOC(k-1)      ,    -V_b(k-1)      ,      -V_c(k-1)
            b = np.array([df.loc[i,'I_r'], self.T_0*df.loc[i,'dK'], 0, 0, 0, 0, self.V_0, 0, -df.loc[i-1,'SOC'], -df.loc[i-1,'V_b'], -df.loc[i-1,'V_c'] ])

            # Deal with non-constant time steps 
            A_temp = A.copy()
            A_temp[ 8,10] = dt # dt * \dot{SOC} 
            A_temp[ 9, 8] = dt # dt * \dot{V_b} 
            A_temp[10, 9] = dt # dt * \dot{V_c}   

            x = np.linalg.solve(A_temp,b)  # Equivalent to matlab A \ b  solves Ax=B for x

            # Integrating charge inefficiency: Propose to solve in two steps:
            # - Solve 100% efficient problem, and see whether charge flows into or out of battery
            #   - If dSOC >(0+tol), re-solve with charging inefficiency
            #   - If dSOC <(0-tol), re-solve with charging inefficency 
            #   - Else assume dSOC==0; do nothing and save results

            tol = 1e-6  # Assumed solver tolerance in matrix factorization, used to determine whether dSOC==0    

            if( x[-1] > tol):  # SOC is increasing, so re-solve with efficiency loss
                A_temp = A_temp.copy()
                A_temp[7,1] = A_temp[7,1]*self.eff  # Redefine the term 1/Q to be eff/Q
                x = np.linalg.solve(A_temp,b)  # Equivalent to matlab A \ b  solves Ax=B for x
            if( x[-1] < tol):  # SOC is decreasing, 
                A_temp = A_temp.copy()
                A_temp[7,1] = A_temp[7,1]/self.eff  # Redefine 1/Q to be 1/(Q*eff)
                x = np.linalg.solve(A_temp,b)  # Equivalent to matlab A \ b  solves Ax=B for x

            for j in range(0,len(x)):
                df.loc[i, self.variables[j]] = x[j]

            if ( not(self.isFeasible( df.loc[i,:], self.constraints, self.variables)) & feasible):
                feasible = False  # This will only be lowered once
                failStep = i

            if ( (not feasible) & stopOnError ):  # If it is not feasible and we are not ignoring constraints, 
                break
        
        if feasible:  # If we think the result is feasible, need to check whether the end state and start state are close to each other
            tol = 0.05  # Relative tolerance for landing within the initial state
            dSOC = (initVariables['SOC'] - df.loc[i,'SOC'])/initVariables['SOC']
            dV_b = (initVariables['V_b'] - df.loc[i,'V_b'])/initVariables['V_b']
            dV_c = (initVariables['V_c'] - df.loc[i,'V_c'])/initVariables['V_c']

            if (abs(dSOC) > tol) | (abs(dV_b) > tol) | (abs(dV_c) > tol):   # if we violate the tolerance
                feasible = False
                failStep = i

        if returnDf:
            return (feasible,df,failStep)
        else:
            return feasible
    
    def isFeasible(self, dataSeries, constraintSet,variables):
        # ConstraintSet is a Pandas dataframe with rows 'max' and 'min', and variables in each column
        isMinFeasible = np.all(dataSeries[variables] >= constraintSet.loc['min',variables])
        isMaxFeasible = np.all(dataSeries[variables] <= constraintSet.loc['max',variables])
        return (isMinFeasible & isMaxFeasible)