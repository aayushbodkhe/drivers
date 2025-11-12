# ----- Fuzzy Sets -----
X = [1, 2, 3, 4, 5]
A = {1: 0.2, 2: 0.7, 3: 1.0, 4: 0.4, 5: 0.1}
B = {1: 0.6, 2: 0.2, 3: 0.9, 4: 0.3, 5: 0.5}

# ----- Basic Operations -----
union        = {x: max(A[x], B[x]) for x in X}
intersection = {x: min(A[x], B[x]) for x in X}
complement_A = {x: 1 - A[x] for x in X}

print("Union:", union)
print("Intersection:", intersection)
print("Complement of A:", complement_A)

# ----- Cartesian Product (A × B and B × A) -----
R1 = {(x, y): min(A[x], B[y]) for x in X for y in X}
R2 = {(x, y): min(B[x], A[y]) for x in X for y in X}

print("\nFuzzy Relation R1 (A × B):")
for p, v in R1.items(): print(p, ":", v)

print("\nFuzzy Relation R2 (B × A):")
for p, v in R2.items(): print(p, ":", v)

# ----- Max-Min Composition -----
Comp = {(x, z): max(min(R1[(x, y)], R2[(y, z)]) for y in X) for x in X for z in X}

print("\nMax-Min Composition (R1 ∘ R2):")
for p, v in Comp.items(): print(p, ":", v)

X=[1,2,3,4,5]
A={1:0.2,2:0.7,3:1.0,4:0.4,5:0.1}
B={1:0.4,2:0.3,3:0.2,4:0.5,5:0.7}

union={x:max(A[x],B[x]) for x in X}
intersection={x:min(A[x],B[x]) for x in X}
comple_A={x:x-1-A[x] for x in X}
print("union",union)
print("intersection",intersection)
print("complement", comple_A)

r1={(x,y):min(A[x],B[y]) for x in X for y in X}
r2={(x,y):min(B[x],A[y]) for x in X for y in X}

for p,v in r1.items():print(p,":",v)
for p,v in r2.items():print(p,":",v)

com={(x,z):max(min(r1[(x,y)],r2[(y,z)])for y in X)for x in X for z in X}
for p,v in com.items():print(p,":",v)