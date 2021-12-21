'''
clst = []
Anslst =[]
Set = {1,2,3}
SetL=  list(Set)
for i in SetL:
    a = {i}
    clst.append(a)
Anslst = clst.copy()

# n1 = set()
for k in clst:
    n = k.copy()
    for h in range(len(SetL)):
        n.add(SetL[h])
        if n in Anslst:
            n = k.copy()
        elif n not in Anslst:
            Anslst.append(n.copy())

Anslst.append({})

print(Anslst)
'''
'''
def Subset(Set):
    clst = []
    # Anslst =[]

    SetL=  list(Set)
    for i in SetL:
        a = {i}
        clst.append(a)
    Anslst = clst.copy()

# n1 = set()
    for k in clst:
        n = k.copy()
        for h in range(len(SetL)):
            n.add(SetL[h])
            if n in Anslst:
                n = k.copy()
            elif n not in Anslst:
                Anslst.append(n.copy())

    Anslst.append({})
    return Anslst

A= Subset({1,2,3,4})
if  {3,4,1} in A: #{2,4} or
    print(True)
else:
    print(False)

print(A)
print(len(A))

Anslst = []
SetRun = set()
Set1 = {1,2,3,4}
Setlst1 = list(Set1)

for i in range(len(Set1)):
    for i1 in range(i+1):
         # for R in Set1:
         for k in Set1:
             if i1 != i+1 and len(Anslst) != 0:
                 CopSet2 = Anslst[-1].copy()
                 CopSet2.add(k)
                 if CopSet2.copy() in Anslst:
                     pass
                 elif CopSet2.copy() not in Anslst:
                    Anslst.append(CopSet2.copy())
             else:
                SetRun.add(k)
                Setcopy = SetRun.copy()
                if Setcopy in Anslst:
                    SetRun.clear()
                elif Setcopy not in Anslst:
                    Anslst.append(Setcopy)
                    SetRun.clear()

print(Anslst)

Set2  = { 1 ,2,3,4 , 5}
Anslst = [{1} , {2} ,{3} , {4} , {5}]
Decre = [4,3,2,1]
LstSetU = []
for i in Decre:
    for i1 in Anslst:
        if len(i1) == len(Set2) - i:
            LstSetU.append(i1)
        else:
            pass
    for i3 in LstSetU:
        for i4 in Set2:
            Newset=  i3.copy()
            Newset.add(i4)
            if Newset in Anslst:
                pass
            elif Newset not in Anslst:
                Anslst.append(Newset)

Anslst.append({})
print(Anslst)
print(len(Anslst))
'''

def CartesianProduct(Set1 , Set2):
  try:
    lst1 = []
    lst2 = []
    SetAns = set()
    lsttest = []
    for i in Set1:
      lst1.append(i)
    for k in Set2:
      lst2.append(k)
    for h in Set1:
      for d in Set2:
        Tup = (h,d)
        SetAns.add(Tup)
    return SetAns
  except(RuntimeError , ValueError):
    return "Error"


def Subset(Set):
    Anslst= []
    Decre = []
    LstSetU = []
    for l in Set:
        a = {l}
        Anslst.append(a)
    for l2 in range(len(Set) + 1):
        Decre.append(l2)
    Decre.reverse()
    Decre.remove(0)
    for i in Decre:
        for i1 in Anslst:
            if len(i1) == len(Set) - i:
                LstSetU.append(i1)
            else:
                pass
        for i3 in LstSetU:
            for i4 in Set:
                Newset=  i3.copy()
                Newset.add(i4)
                if Newset in Anslst:
                    pass
                elif Newset not in Anslst:
                    Anslst.append(Newset)

    Anslst.append({})
    return Anslst


if __name__ == '__main__':

    Set3 = {"a" , "b" , "c"}
    print(Subset(Set3))
    print(len(Subset(Set3)))

    A= {3,5,6,7}
    B = {6,10,12,14}
    Carset  = CartesianProduct(A , B)
    # Carset = Subset(CartesianProduct(A , B))
    # print(Carset)
    # print(len(Carset))

'''
empset = set()
for i in Carset:
    for (x,y) in i:
        if x % y == 0:
            # print((x , y))
            empset.add((x,y))
        else:
            pass
print(empset)
'''
'''
    empset = set()
    for (x,y) in Carset:
        if y%x == 0:
            empset.add((x ,y))
        else:
            pass
    print(empset)
    print(empset.issubset(Carset))

    print(len(CartesianProduct({1,2,3} , {1,2,3})) + len(CartesianProduct({4,5,6,7},{1,2,3})))

    a = CartesianProduct({-3,0,4} , {9 ,5 ,-5, 8}) - CartesianProduct({1,12,20} , {9,5,-5,8})
    print(len(a))
'''
