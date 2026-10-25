import streamlit as st
import pandas as pd
import random

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TV Scheduling - Genetic Algorithm",
    page_icon="📺",
    layout="wide"
)

st.title("📺 TV Program Scheduling using Genetic Algorithm")
st.markdown("""
This application uses a **Genetic Algorithm (GA)** to find an optimal TV program schedule 
that maximizes overall viewer ratings across all available time slots.
""")

# -----------------------------------------------------------------------------
# 2. FILE UPLOADER & DATA PARSING (FIXED KEYERROR ISSUE)
# -----------------------------------------------------------------------------
st.sidebar.header("📁 1. Load Rating Data")
uploaded_file = st.sidebar.file_uploader("Upload 'program_ratings.csv'", type=["csv"])

@st.cache_data
def load_ratings(file):
    program_ratings = {}
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        # Using .iloc[0] and .iloc[1:] fixes the Pandas KeyError
        program = row.iloc[0]
        ratings = [float(x) for x in row.iloc[1:]]
        program_ratings[program] = ratings
    return program_ratings, df

if uploaded_file is not None:
    ratings_dict, df_raw = load_ratings(uploaded_file)
    st.sidebar.success("CSV file loaded successfully!")
else:
    st.info("👈 Please upload your `program_ratings.csv` file in the sidebar to begin.")
    st.stop()

with st.expander("👀 View Loaded Program Ratings Data"):
    st.dataframe(df_raw, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. GA PARAMETERS SETUP (ASSIGNMENT REQUIREMENTS)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 2. GA Parameters Setup")

# Requirement: CO_R slider (Default: 0.8, range: 0.0 to 0.95)
co_r = st.sidebar.slider(
    "Crossover Rate (CO_R)",
    min_value=0.0,
    max_value=0.95,
    value=0.80,
    step=0.01,
    help="Probability of crossover between parents (Default: 0.80)"
)

# Requirement: MUT_R slider (Default: 0.02, range: 0.01 to 0.05)
mut_r = st.sidebar.slider(
    "Mutation Rate (MUT_R)",
    min_value=0.01,
    max_value=0.05,
    value=0.02,
    step=0.005,
    help="Probability of mutating a schedule slot (Default: 0.02)"
)

generations = st.sidebar.number_input("Generations", min_value=10, max_value=1000, value=100, step=10)
pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=200, value=50, step=5)
elitism_size = 2

all_programs = list(ratings_dict.keys())
time_slots = list(range(6, 6 + len(next(iter(ratings_dict.values())))))

# -----------------------------------------------------------------------------
# 4. GENETIC ALGORITHM CORE FUNCTIONS
# -----------------------------------------------------------------------------
def fitness_function(schedule):
    """Calculates total ratings score for a given schedule."""
    total_rating = 0
    for idx, program in enumerate(schedule):
        total_rating += ratings_dict[program][idx]
    return total_rating

def crossover(schedule1, schedule2):
    """Single-point crossover between two schedules."""
    if len(schedule1) < 2:
        return schedule1.copy(), schedule2.copy()
    crossover_point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    """Randomly mutates one program slot in the schedule."""
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, len(schedule_copy) - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def run_genetic_algorithm(gen, pop_s, crossover_rate, mutation_rate, elitism_s):
    """Executes the Genetic Algorithm over specified generations."""
    # 1. Initialize random population
    population = []
    for _ in range(pop_s):
        random_schedule = [random.choice(all_programs) for _ in range(len(time_slots))]
        population.append(random_schedule)

    # 2. Evolutionary loop
    for _ in range(gen):
        new_population = []
        
        # Elitism: retain top individuals
        population.sort(key=lambda s: fitness_function(s), reverse=True)
        new_population.extend(population[:elitism_s])

        # Generate offspring
        while len(new_population) < pop_s:
            parent1, parent2 = random.choices(population, k=2)
            
            # Crossover step
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation step
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:pop_s]

    population.sort(key=lambda s: fitness_function(s), reverse=True)
    best_schedule = population[0]
    best_fitness = fitness_function(best_schedule)
    return best_schedule, best_fitness

# -----------------------------------------------------------------------------
# 5. EXECUTION & DISPLAY RESULTS IN TABLE FORMAT
# -----------------------------------------------------------------------------
if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Optimizing schedule with Genetic Algorithm..."):
        best_sched, max_fitness = run_genetic_algorithm(
            generations, pop_size, co_r, mut_r, elitism_size
        )
    
    st.success("Optimization Completed Successfully!")

    # Display Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Crossover Rate (CO_R)", f"{co_r}")
    m2.metric("Selected Mutation Rate (MUT_R)", f"{mut_r}")
    m3.metric("Total Optimal Rating (Fitness)", f"{max_fitness:.2f}")

    # Build Schedule DataFrame
    results = []
    for idx, prog in enumerate(best_sched):
        results.append({
            "Time Slot": f"{time_slots[idx]:02d}:00",
            "Scheduled Program": prog,
            "Rating": f"{ratings_dict[prog][idx]:.2f}"
        })
    
    schedule_df = pd.DataFrame(results)

    # Display schedule in table format
    st.subheader("📋 Final Optimized TV Schedule")
    st.table(schedule_df)
