import math,random,itertools,decimal

class combination_finder:
    """
    Adopted from pypi library: Combinadics
    Link: https://pypi.org/project/combinadics/
    """
    def __init__(self):
        self.explored_combinations_real=dict()
    
    def LargestV_binary(self,a, b, x):
        """
        Return largest value v where v < a and  Choose(v,b) <= x.
        For example, if a = 8, b = 4 and x = 7, then v = 5 because
        5 < 8 and Choose (5,4)=5 <=7.
        Parameters
        ----------
        a : Integer
            DESCRIPTION.
        b : Integer
            DESCRIPTION.
        x : Integer
            DESCRIPTION.
        Returns
        -------
        Integer.
        """
        v = a-1

        ## Binary search to find the largest v that satisfies this.
        ## Goal: Find the largest value v where v < a and  Choose(v,b) <= x.

        not_found=True

            
        def satisfies_goal(v):
            c=self.reduced_comb(v,b)
    
            return c>x

        v_range=[0,v]
        
        v_half= math.ceil((v_range[1]+v_range[0])/2)

        while v_range[1]-v_range[0]>1:
            if satisfies_goal(v_half): #We look if the left half of the binary list has the needed element.
                v_range=[v_range[0],v_half]
            else:
                v_range=[v_half,v_range[1]]
            v_half= math.floor((v_range[0]+v_range[1])/2)
        if satisfies_goal(v_range[1]):
            return v_range[0]
        else:
            return v_range[1]
        
    def LargestV_bin_search(self,a, b, x):
        """
        Return largest value v where v < a and  Choose(v,b) <= x.
        For example, if a = 8, b = 4 and x = 7, then v = 5 because
        5 < 8 and Choose (5,4)=5 <=7.
        Parameters
        ----------
        a : Integer
            DESCRIPTION.
        b : Integer
            DESCRIPTION.
        x : Integer
            DESCRIPTION.
        Returns
        -------
        Integer.
        """
        v = a-1

        ## Binary search to find the largest v that satisfies this.
        ## Goal: Find the largest value v where v < a and  Choose(v,b) <= x.
            
        def satisfies_goal(v):
            c=self.reduced_comb(v,b)
            return c<x

        v_range=[0,v]
        
        while v_range[1]-v_range[0]>1:
            v_eval_points=np.linspace(v_range[0],v_range[1],bin_size)
            for point in enumerate(reversed(v_eval_points)):
                if satisfies_goal(point):
                    v_range=[point,v_range[-i-1]]

        if satisfies_goal(v_range[1]):
            return v_range[1]
        else:
            return v_range[0]

    def LargestV(self,a, b, x):
        """
        Return largest value v where v < a and  Choose(v,b) <= x.
        For example, if a = 8, b = 4 and x = 7, then v = 5 because
        5 < 8 and Choose (5,4)=5 <=7.
        Parameters
        ----------
        a : Integer
            DESCRIPTION.
        b : Integer
            DESCRIPTION.
        x : Integer
            DESCRIPTION.
        Returns
        -------
        Integer.
        """
        v = a-1
        c = self.reduced_comb(v,b)
        while (c > x):
            v = v-1
            c = self.reduced_comb(v,b)
        return v
    
    def reduced_comb(self,n,k):
        if (n,k) in self.explored_combinations_real.keys():
            return self.explored_combinations_real[(n,k)]
        else:
            try:
                c=math.comb(n,k)
            except:
                print(n,k)
                raise Exception('check')
            self.explored_combinations_real[(n,k)]=c
            return c
    
    def find_elm(self,n,k,m):
        """
        n>k
        """
        
        if k>n:
            raise Exception('Not possible')
        ans = []

        a = n
        b = k
        # x is the "dual" of m, duals sum to Choose(n,k) - 1
        x = (self.reduced_comb(n, k) - 1) - m

        for i in range(k):
            # largest value v, where v < a and vCb < x
            try:
                if x<self.reduced_comb(a,b)/10 or n-a>10:
                    try:
                        ans.append(self.LargestV_binary(a, b, x))
                    except:
                        print(a,b,x)
                        raise Exception('check previous')
                else:
                    ans.append(self.LargestV(a, b, x))
            except OverflowError:
                if x<int(decimal.Decimal(self.reduced_comb(a,b))/10) or n-a>10:
                    ans.append(self.LargestV_binary(a, b, x))
                else:
                    ans.append(self.LargestV(a, b, x))
                
                
            x = x - self.reduced_comb(ans[i], b)
            a = ans[i]
            b = b-1

        for i in range(k):
            ans[i] = (n-1) - ans[i]

        return ans