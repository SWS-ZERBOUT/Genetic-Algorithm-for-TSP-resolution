import streamlit as st
import numpy as np
import random, operator
import matplotlib.pyplot as plt
import random




st.title('Traveling Salesman Solver')
st.header('based on genetic algorithm best approximate solution')


with st.sidebar:

    nbr_villes = st.number_input(
        "Nombre de Villes", min_value=10, max_value=100, step=10
    )
    population_taille = st.number_input(
        "Taille population", min_value=10, max_value=100, step=10
    )
    nbr_generations = st.number_input(
        "Nombre de Generation", min_value=100, max_value=10000, step=100
    )
st.header("Your Data")
st.write("""
 le nombre de villes donné
""",nbr_villes)
st.write("""
 la taille de la population donnée
""",population_taille)
st.write("""
 le nombre de generations donné 
""",nbr_generations)


population = []
x=np.random.uniform(0,1,nbr_villes)
y=np.random.uniform(0,1,nbr_villes)
chemin = np.arange(nbr_villes)




cityList = []
for i in range(0,nbr_villes):
    cityList.append((x[i],y[i]))



def creer_nv_individu(n_villes):

    pop=set(np.arange(n_villes,dtype=int))
    route=list(random.sample(pop,n_villes))
    for i in range(len(route)):
        if route[i] == 0:
            route = np.roll(route, -i, axis=None)
            
    return route

def cree_population_initial(taille,n_villes):
    population = []
    
    for i in range(0,taille):
        population.append(creer_nv_individu(n_villes))
        
    return population

population_initiale = cree_population_initial(population_taille,nbr_villes)







    
#calcul distance
def distance(i,j):
    return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)

#
def fitness(route,CityList):
    score=0  
    for i in range(1,len(route)):
        k=int(route[i-1])
        l=int(route[i])

        score = score + distance(CityList[k],CityList[l]) 
    score = score + distance(cityList[route[-1]],cityList[route[0]])    
    return score

#
def score_population(population, CityList):  
    scores = [] 
    for i in population:
        scores.append(fitness(i, CityList))
    return scores
#
score = []
score = score_population(population_initiale, cityList)
#
population_fitness = 0
for i in range(population_taille):
    population_fitness = population_fitness + score[i]
#st.write(population_fitness)
#
chromosome_probabilities1 = [(fitness(route,cityList))/population_fitness for route in population_initiale]

chromosome_probabilities2 = [(1-chromosome_probabilities1[i])/(population_taille-1) for i in range(population_taille)]
# selection
import numpy.random as npr
def selectOne(population):
    max = sum([fitness(c,cityList) for c in population])

    selection_prob = [fitness(c,cityList)/max for c in population]
    selection_probs = [(1-selection_prob[i])/(population_taille-1) for i in range(population_taille)]
    return population[npr.choice(len(population), p=selection_probs)]
def selection(population):
    selected = []
    for i in range(int(len(population)/2)):
        selected.append(selectOne(population))
    return selected

# croisement
def crossover(a,b):

    child=[]
    childA=[]
    childB=[]

    geneA=int(random.random()* len(a))
    geneB=int(random.random()* len(b))
    
    start_gene=min(geneA,geneB)
    end_gene=max(geneA,geneB)
    
    for i in range(start_gene,end_gene):
        childA.append(a[i])
       
    childB=[item for item in a if item not in childA]
    child=childA+childB
   
    return child

def breedPopulation(mating_pool):
    children=[]
    for i in range(len(mating_pool)-1):
                children.append(crossover(mating_pool[i],mating_pool[i+1]))
    return children

# mutation

def mutate(route,probablity):

    route=np.array(route)
    for swaping_p in range(len(route)):
        if(random.random() < probablity):
            swapedWith = np.random.randint(0,len(route))
            
            temp1=route[swaping_p]
            
            temp2=route[swapedWith]
            route[swapedWith]=temp1
            route[swaping_p]=temp2
    
    return route

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# Classement

def rankRoutes(population,City_List):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(population[i],City_List)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)

def best_solution(population,city_List):
    ranked_population = rankRoutes(population,city_List)
    i = ranked_population[0][0]
    solBest = population[i]
    return solBest




# Algorithme Génétique

def nextGeneration(CityList,currentGen, mutationRate):
    selectionResults = selection(currentGen)
 
    children = breedPopulation(selectionResults)
   
    nextGeneration = mutatePopulation(children, mutationRate)
    nextGeneration = nextGeneration + selectionResults 
   
    nextGeneration.append(best_solution(currentGen,CityList))
    return nextGeneration


def geneticAlgorithm(CityList,mutationRate,generations):
    bestfitness_pergen = []
    gen = []
    population = population_initiale
    st.header("Results")
    st.write("Meilleur route de la population intiale :",best_solution(population_initiale,CityList),"sa distance :",fitness(best_solution(population_initiale,CityList),CityList))
    
    fig, ax = plt.subplots()

    ax.plot(x[best_solution(population_initiale,CityList)],y[best_solution(population_initiale,CityList)], marker= 'o', color='r')
    ax.plot([x[best_solution(population_initiale,CityList)[-1]],x[best_solution(population_initiale,CityList)[0]]],[y[best_solution(population_initiale,CityList)[-1]],y[best_solution(population_initiale,CityList)[0]]])
    fig.suptitle("Meilleur solution de la population initiale")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    st.write(fig)

    bestfitness_pergen.append(fitness(best_solution(population_initiale,CityList),CityList))
    for i in range(0, generations):
        population = nextGeneration(CityList,population, mutationRate)
        bestfitness_pergen.append(fitness(best_solution(population,CityList),CityList))
        gen.append(i+1)

    route = best_solution(population,CityList)
    for i in range(len(route)):
        if route[i] == 0:
            route = np.roll(route, -i, axis=None)

    st.write("Meilleur route de la dernière population :",route,"sa distance :",fitness(route,CityList))

    #####

    fig2, ax = plt.subplots()
    ax.plot(x[route],y[route], marker= 'o', color='g')
    ax.plot([x[route[-1]],x[route[0]]],[y[route[-1]],y[route[0]]])
    fig2.suptitle('Meilleur route de la dernière population')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')

    st.write(fig2)
    

    st.header("Distance per generation")
    fig1, ax = plt.subplots()
    fig1.suptitle("Developpement de la distance par génération")
    
    ax.plot(bestfitness_pergen)
    ax.set_ylabel('Distance')
    ax.set_xlabel('Generation')
    st.write(fig1)



#Solution

geneticAlgorithm(cityList,0.01,nbr_generations)
