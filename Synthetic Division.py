class syntheticDivision:
    def __init__(self, Coef = list):
        try:
            count = len(Coef)
            self.count = count
            self.Coef = Coef
        except(ValueError, TypeError,RuntimeError):
            return "Sorry there has some Problem at __init__"

    def Factor(self, wanted):
        try:
            Coef = self.Coef
            k = []
            m= []
            a0 = Coef[-1]
            an = Coef[0]
            km = []
            if a0 < 0 :
                a0 = -1*a0
            elif a0 > 0 :
                a0 = a0
            elif a0 == 0:
                pass
            for i in range(1 ,((a0)+1)):
                if a0 % i == 0:
                    k.append(i)
                    i1 = -1 * i
                    k.append(i1)
                elif a0 % i != 0:
                    pass

            for i in range(1 , (an +1)):
                 if an % i == 0:
                    m.append(i)
                    i2 = -1 * i
                    m.append(i2)
                 elif an % i != 0:
                    pass

            for i in m:
                for i3 in k:
                    Anskm = i3 / i
                    if Anskm not in km :
                        km.append(Anskm)
                    elif Anskm in km:
                        pass

            if wanted == "k":
                return k
            elif wanted == "m":
                return m
            elif wanted == "km":
                return km


        except(RuntimeError, ZeroDivisionError,TypeError):
            return "Problem on operation"

    def Synthetic(self):
        try:
            Coef = self.Coef
            count = self.count
            km = self.Factor("km")
            if count == 3:
                round = 1
            elif count != 3:
                round = count- 3

            Ans = [] #A = True for checking that we have find the answer
            #PartAns= []
            #lstmul = []
            i_lst = []
            #collect = []
            kmcheck = []
            Coefcheck = []
            check = False
            for c in range(round):
                c = 0
                while True:
                    for i in km:
                        for i1 in Coef:
                            Coefcheck.append(i1)
                            if check == False:
                                n1 = 0
                            elif check == True:
                                pass
                            n2 = i1 + n1 #Important
                            if len(Coefcheck) != len(Coef): #i1 != Coef[-1]:
                                c1 = i * n2
                                n1 = c1
                            elif len(Coefcheck) == len(Coef):
                                pass
                            check = True
                            Ans.append(n2)

                        kmcheck.append(i)

                        if len(Ans) == len(Coef):

                            if Ans[-1] == 0 and c == round-1:
                                i_lst.append(i)
                                Ans.append(i_lst)
                                check = False
                                return Ans
                                break

                            elif Ans[-1] != 0 and len(kmcheck) == len(km):
                                kmcheck.clear()
                                return "No Answer by division synthesis"

                            elif Ans[-1] != 0:
                                Ans.clear()
                                Coefcheck.clear()
                                check = False



                            elif Ans[-1] == 0 and c != round - 1:
                                Ans.pop(-1)
                                AnsCopy = Ans.copy()
                                Coef = AnsCopy
                                Ans.clear()
                                check = False
                                km.append(i)
                                i_lst.append(i)
                                c += 1

                        else:
                            pass




        except(ValueError,RuntimeError, ZeroDivisionError,TypeError,MemoryError):
            return "Problem during synthesis"


if __name__ == '__main__':
    #print("Test and wish")
    # obj = syntheticDivision([2,-2.5,-4,3,2])
    #obj = syntheticDivision([1,4,4])
    obj = syntheticDivision([9,6,1])
    #obj = syntheticDivision([2,5 ,4 ,2])
    #print(obj.Factor("km"))
    print(obj.Synthetic())
