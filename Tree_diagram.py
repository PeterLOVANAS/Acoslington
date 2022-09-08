import itertools
import math
import numpy as np
#for i in itertools.permutations("ABCD"):
#    print(i)

class Tree_Diagram:
    def __init__(self , text):
        self.text = text


    def Exponential(self , k = int):
        anslst = []
        fortup = []
        text = self.text

        lsta =  list(text)
        lst =  list(map(list , text)) # Pick each individual alphabet in text and make it to list like ["a"] and later they put the output of the map() into another list


        #lsta = ["a" , "b", "c", "d"]
        #lst = [["a"] , ["b"] , ["c"] , ["d"]]
        storage = lst.copy()
        stor2 = []
        #for i in lsta:
        for n in range(k-1):
            #length = len(storage)
            for g in storage:
                for h in lsta:
                    g.append(h)
                    stor2.append(g.copy())
                    g.pop(-1)
            storage = stor2.copy()
            stor2.clear()

        for l in storage.copy():
            anslst.append(tuple(l))

        return anslst
            #storage.clear()

    def Permutations(self , k = int):
        anslst = []

        text = self.text

        if "," in text:
            lsta = text.split(",")
            lsta2 = text.split(",")
            lst = []
            for i in text.split(","):
                lst.append([i])

        elif "," not in text:
            lsta =  list(text)
            lsta2 = list(text)
            lst = list(map(list , text)) # Pick each individual alphabet in text and make it to list like ["a"] and later they put the output of the map() into another list

        storage = lst.copy()
        stor2 = []
        #for i in lsta:
        if k <= len(lsta2):
            pass
        elif k > len(lsta2):
            return "k should smaller or equal to length of string"

        for n in range(k-1):
            #length = len(storage)
            for g in storage:
                if "," in text:
                    lsta2 = text.split(",")
                elif "," not in text:
                    lsta2 = list(text)

                for f in g:
                    lsta.remove(f)
                for h in lsta:
                    g.append(h)
                    stor2.append(g.copy())
                    g.pop(-1)
                lsta = lsta2
            storage = stor2.copy()
            stor2.clear()

        for l in storage.copy():
            anslst.append(tuple(l))

        return anslst

    def Combination(self , k = int):
        ansset = set()
        tuplst = self.Permutations(k)
        ans = list(map(frozenset , tuplst))
        for i in ans:
            ansset.add(i)
        ans1 = list(map(tuple , ansset))
        return ans1

    def Rearrange_Count(self , slst = list):
        text = self.text
        lsttex = text.split(",")
        s1 = set()
        ck1 = self.Permutations(k = len(lsttex))
        if all(isinstance(x , tuple) for x in slst) == True :
            pass
        elif all(isinstance(x , tuple) for x in slst) == False:
            return "Every element in list should be tuples"

        for x2 in slst:
            for x3 in x2:
                if x3 in lsttex:
                    pass
                elif x3 not in lsttex:
                    return "Every element should be in the text"

        for j in slst:
            for j1 in j:
                lsttex.remove(j1)

        for i1 in ck1:
            ans2 = []
            for l in slst:
                lst = []
                for k in l:
                    a = i1.index(k)
                    lst.append(a)
                f = frozenset(lst)
                ans2.append(f.copy())
                lst.clear()
            for x in lsttex:
                ans2.append(i1.index(x))

            ans3 = tuple(ans2)
            #ans1 = (frozenset([i1.index("A1") , i1.index("A2")]) , (i1.index("B") , i1.index("L") , i1.index("G")) ,i1.index("E") , i1.index("R"))
            s1.add(ans3)

        return len(s1)









    def P(self , n = int ,k = int):
        if k <= n:
            return math.factorial(n) / math.factorial(n-k)
        else:
            return None

    def C(self , n = int , k = int):
        if k <= n:
            return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
        else:
            return None

    def show(self , showlst = list):
        for i in showlst:
            print(i)

    def CountShow(self , ele = str , num = int,  lst = list):
        ans = []
        if all(isinstance(x , tuple) for x in lst) == True:
            pass
        elif all(isinstance(x , tuple) for x in lst) == False:
            return None
        for i in lst:
            if i.count(ele) == num:
                ans.append(i)
            elif i.count(ele) != num:
                pass
        return ans

    def Cyclic_Permutation(self , start = str):
        anslst = []
        Permulin = self.Permutations(k = len(self.text))
        if start in self.text:
            pass
        elif start not in self.text:
            return "Start value must be in the permutation"
        for k in Permulin:
            Mapdict = dict(zip(Permulin[0] , k))
            n = start
            lst = []
            lst.append(n)
            #Check = list(Permulin[0])
            while True:
                mapval = Mapdict[n]
                if mapval in lst:
                    break
                elif mapval not in lst:
                    lst.append(mapval)
                n = mapval
            if len(lst) == len(Permulin[0]):
                anslst.append(tuple(lst))
            elif len(lst) != len(Permulin[0]):
                pass
            lst.clear()

        return anslst






        '''
        ini = np.zeros((2, 3))
        stan = np.asarray(Permulin[0])
        ini[0 , :] = stan
        for k in Permulin:
            ini[1,:] = np.asarray(k)
            for j in range(len(self.text)):
                inp = ini[0 , j]
                # Column i , row 1
        '''



obj1 = Tree_Diagram("ABCD")
print(obj1.Exponential(3))
print(len(obj1.Exponential(3)))

print(obj1.Exponential(2))
print(len(obj1.Exponential(2)))

print(obj1.Exponential(1))
print(len(obj1.Exponential(1)))

obj2 = Tree_Diagram("abcd")
print(obj2.Permutations(2))
print(obj2.Permutations(3))
print(len(obj2.Permutations(3)))
print(obj2.Combination(3))
obj2.show(obj2.Permutations(2))
print(obj2.P(4 , 3))
print(obj2.C(4, 3))

obj5 = Tree_Diagram

#binomial test
obj3 = Tree_Diagram("ab")
print(obj3.Exponential(3))
obj4 = Tree_Diagram("123")
print(obj4.Combination(2))
check = obj3.Exponential(3)
print(obj3.CountShow("a" , 2 ,check))

obj5 = Tree_Diagram("A1,L,A2,B,A3,M,A4")
ck = obj5.Permutations(7)
print(len(ck))
obj6 = Tree_Diagram("alabama")
print(obj6.Exponential(2))
obj7= Tree_Diagram("A1,A2,A3,A4")
print((obj7.Permutations(4)))

'''
Find the set of indexes of A in each permutation of ALABAMA. For example, {2,3,5,7} is a set contain the indexes of a 
in that permutation, if someway there have another permutation that have the same set of indexes;therefore, they're equivilance or
just a rearrangement of others.
{frozenset(A) , [ele left]} => we will check both {A} and [ele left]. 
* frozenset(A)  is the set of A indexes  and [ele left] is the list of ordered left element for example, L or B.
We will make this {frozenset(A) , [ele left]} into a frozenset and then put it into a larger {}.  
*Not working ^
'''
# Set can contain Hashable object only (list is not hashable but tuple is)
# When tuple or frozenset want to __hash__ they will check whether every elements inside it is hashable or not
a = (frozenset([2, 1]),  [5,6])
b = (frozenset([2, 1]), [6,5])
c = ([6,5] , frozenset([2, 1]))
d = (frozenset([1,2]),  [5,6])
print(a== b == c)
print(a== b)
#s = {a , b , d}
#print(s)
a1 = (frozenset([2, 1]), (5,6)) #All elements in tuple (hasable object) is hashable
b1 = (frozenset([2, 1]) , (6,5))
d1 = (frozenset([1,2]) , (5,6))
print({a1 , b1 , d1})

s = set()
for i1 in ck:
    ans = (frozenset([i1.index("A1") , i1.index("A2") , i1.index("A3") , i1.index("A4")]) , (i1.index("B") , i1.index("L") , i1.index("M")) )
    s.add(ans)
print(len(s))

s1 = set()
obj8 = Tree_Diagram("A1,L,G,E,B,R,A2")
ck1 = obj8.Permutations(7)
for i1 in ck1:
    ans1 = (frozenset([i1.index("A1") , i1.index("A2")]) , i1.index("B") , i1.index("L") , i1.index("G") ,i1.index("E") , i1.index("R"))
    s1.add(ans1)
print(len(s1))
obj9 = Tree_Diagram("1234")
print(obj9.Cyclic_Permutation("1"))
print("***Test***")
obj10 = Tree_Diagram("A1,L,G,E,B,R,A2")
print(obj10.Rearrange_Count(slst= [("A1" , "A2")]))


s3 = set()
obj11 = Tree_Diagram("M,I1,S1,S2,O,U,R,I2")
ck2 = obj11.Permutations(8)
for i1 in ck2:
    ans1 = (frozenset([i1.index("I1") , i1.index("I2") ]) , frozenset([i1.index("S1") , i1.index("S2")]) , i1.index("O") , i1.index("U") , i1.index("R") , i1.index("M"))
    s3.add(ans1)

#print(len(s3))

print(obj11.Rearrange_Count(slst = [("I1" , "I2") , ("S1" , "S2")]))
obj12 = Tree_Diagram("r1,r2,b1,b2,b3,g")
print(len(obj12.Permutations(k = 6)))
print(obj12.Rearrange_Count(slst = [("r1" , "r2") , ("b1" , "b2" , "b3")]))
obj13 = Tree_Diagram("r1,r2,r3,r4,b1,b2,g1,g2,g3")
print(obj13.Rearrange_Count(slst = [("r1","r2","r3","r4"),("b1","b2"),("g1","g2","g3")]))
obj14 = Tree_Diagram("abcdefgh")
sn = obj14.Permutations(k = 8)
an = []
for i in sn:
    if i[-1] == "a" or i[-1] == "b":
        pass
    else:
        an.append(i)

print(len(an))
an2 = []
for i in sn:
    if i[0] == "c" or i[0] == "d" or i[0] == "e":
        an2.append(i)
    else:
        pass

print(len(an2))

an3 = []
for i in sn:
    if i[0] == "c" or i[0] == "d" or i[0] == "e":
        if i[-1] == "a" or i[-1] == "b":
            pass
        else:
            an3.append(i)
    else:
        pass
print(len(an3))
obj5 = Tree_Diagram("HT")
a = obj5.Exponential(10)
klst= []
hlst= []
copa = a.copy()
#print(a)

for i in a:

    for k in enumerate(i):
        klst.append(k)

    for k1 in klst:
        if k1[1] == "H":
            hlst.append(k1[0])
        else:
            pass

    for k2 in hlst:
        if len(hlst) >= 2:

            if hlst[-1] != k2 and k2 + 1 == hlst[hlst.index(k2) + 1]:
                copa.remove(i)
                break
            else:
                pass
        else:
            break

    klst.clear()
    hlst.clear()


#print(copa)
print(len(copa))





'''
tls = []
for it in s1:
    if isinstance(it[1] , frozenset) == False:
       tls.append(it)
    else:
        pass
print(len(s1))
print("***")
print(len(tls))
'''
'''
for it in s1:
    if isinstance(it[1] , frozenset) == True:
       print(it)
    else:
        pass
'''
'''
Deep into the tuple => (frozenset(t.index(A)) , (t.index(ele_left)))
1.indexes of each A in frozenset can be interchange, for example, A1 can interchange the index with A3 but anyway the set of index is the same, {1,2,3,4} == {3,2,1,4}
2.the second tuple contain indexes of other character that are not identical when be rearrange with each other so, (1,2,5) != (2,1,5) 
3.When put into as set. The one that are identical will be erased from the set
*Working! ^
'''
obj15 = Tree_Diagram("123456")
k  = obj15.Exponential(2)
print(k)


lst = []
for i in k:
    if "1" != i[0]:
        lst.append(i)
    else:
        pass
print(lst)

s = 0
for i1 in lst:
    s += sum(tuple(map(int,i1)))

print(s)


