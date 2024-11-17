import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt
import random

def load_and_clean_data(file_path):
    """
    Load and clean the railway dataset
    """
    try:
        # Read the CSV file with low_memory=False to handle mixed types
        df = pd.read_csv(file_path, low_memory=False)
        
        # Clean the data
        df = df.dropna(subset=['Train No', 'Station Code', 'Distance'])
        
        # Ensure Distance is numeric
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
        
        # Convert SEQ to numeric, coerce errors to NaN
        df['SEQ'] = pd.to_numeric(df['SEQ'], errors='coerce')
        
        # Drop rows where conversion to numeric failed
        df = df.dropna(subset=['Distance', 'SEQ'])
        
        # Convert time columns to datetime
        for col in ['Arrival time', 'Departure Time']:
            try:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time
            except:
                print(f"Warning: Could not convert {col} to time format")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features_for_forecasting(df):
    """
    Prepare features for demand forecasting
    """
    try:
        # Create station-wise features
        station_features = pd.DataFrame()
        
        # Count number of trains per station
        train_counts = df.groupby('Station Code')['Train No'].nunique().reset_index()
        station_features['Station Code'] = train_counts['Station Code']
        station_features['train_count'] = train_counts['Train No']
        
        # Calculate average distance (only for numeric values)
        distance_avg = df.groupby('Station Code')['Distance'].mean().reset_index()
        station_features = station_features.merge(distance_avg, on='Station Code', how='left')
        
        # Calculate average sequence number (only for numeric values)
        seq_avg = df.groupby('Station Code')['SEQ'].mean().reset_index()
        station_features = station_features.merge(seq_avg, on='Station Code', how='left')
        
        # Add junction indicator
        station_features['is_junction'] = df.groupby('Station Code')['Station Name'].first().str.contains('JN').astype(int).reset_index()['Station Name']
        
        # Fill any remaining NaN values with 0
        station_features = station_features.fillna(0)
        
        return station_features
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None

def train_demand_model(features, target):
    """
    Train a Random Forest model for demand forecasting
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features.drop(['Station Code'], axis=1),
            target,
            test_size=0.2,
            random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate MAE
        mae = mean_absolute_error(y_test, y_pred)
        
        return model, scaler, mae, X_test, y_test, y_pred
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None, None, None, None


def extract_connected_subgraph(df, start_station, end_station, num_nodes=20):
    """
    Extract a connected subgraph containing start and end stations
    with approximately num_nodes stations
    """
    try:
        # Create a NetworkX graph from the railway network
        G = nx.Graph()

        # Add edges from sequential stations in train routes
        for _, group in df.groupby('Train No'):
            sorted_stations = group.sort_values('SEQ')
            for i in range(len(sorted_stations) - 1):
                station1 = sorted_stations.iloc[i]['Station Code']
                station2 = sorted_stations.iloc[i + 1]['Station Code']
                distance = float(sorted_stations.iloc[i + 1]['Distance']) - float(sorted_stations.iloc[i]['Distance'])
                G.add_edge(station1, station2, weight=distance)

        # Ensure start and end stations are in the graph
        if start_station not in G or end_station not in G:
            print("Start or end station not found in graph")
            return None, None

        # Find shortest path between start and end stations
        try:
            path_stations = nx.shortest_path(G, start_station, end_station, weight='weight')
        except nx.NetworkXNoPath:
            print("No path exists between start and end stations")
            return None, None

        # Initialize subgraph with stations in the shortest path
        selected_stations = set(path_stations)

        # Add additional stations to reach desired size
        while len(selected_stations) < num_nodes and len(selected_stations) < len(G.nodes):
            # Get all neighbors of current selected stations
            neighbors = set()
            for station in selected_stations:
                neighbors.update(G.neighbors(station))

            # Remove already selected stations from neighbors
            neighbors -= selected_stations

            if not neighbors:
                break

            # Select a random neighbor that's part of some train route
            new_station = random.choice(list(neighbors))
            selected_stations.add(new_station)

        # Extract the subgraph
        subgraph = G.subgraph(selected_stations)

        # Convert to list and create station mapping
        stations_list = list(subgraph.nodes())
        station_mapping = {station: idx for idx, station in enumerate(stations_list)}

        return stations_list, subgraph

    except Exception as e:
        print(f"Error extracting subgraph: {e}")
        return None, None

def create_distance_matrix_from_subgraph(subgraph, stations_list):
    """
    Create a distance matrix from the NetworkX subgraph
    """
    try:
        n = len(stations_list)
        distances = np.full((n, n), np.inf)

        # Fill the distance matrix using edge weights from the subgraph
        for i in range(n):
            for j in range(n):
                if i == j:
                    distances[i][j] = 0
                else:
                    station1 = stations_list[i]
                    station2 = stations_list[j]
                    try:
                        # Get shortest path length between stations
                        distance = nx.shortest_path_length(
                            subgraph, 
                            station1, 
                            station2, 
                            weight='weight'
                        )
                        distances[i][j] = distance
                        distances[j][i] = distance
                    except nx.NetworkXNoPath:
                        continue

        return distances

    except Exception as e:
        print(f"Error creating distance matrix: {e}")
        return None

def optimize_path(distance_matrix, start_idx, end_idx):
    """
    Optimize path using OR-Tools with correct initialization
    """
    try:
        if distance_matrix is None:
            return None

        # Create the routing index manager with proper initialization
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),  # Size of the distance matrix
            1,                     # Number of vehicles (1 for single path)
            int(start_idx)         # Start index (depot)
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convert to integers for OR-Tools

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            path = []
            while not routing.IsEnd(index):
                path.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            path.append(manager.IndexToNode(index))
            return path
        return None
    except Exception as e:
        print(f"Error optimizing path: {e}")
        return None
def get_original_path(df, start_station, end_station, stations_list):
    """
    Get the original path distance considering all stations in stations_list
    similar to a TSP approach.
    """
    try:
        # Find a train that visits all stations in the subgraph (if possible)
        for train in df['Train No'].unique():
            train_route = df[df['Train No'] == train].sort_values('SEQ')
            if all(station in train_route['Station Code'].values for station in stations_list):
                # Calculate the total distance traveling through all stations in the subgraph
                total_distance = 0
                path = []
                for i in range(len(stations_list)):
                    current_station = stations_list[i]
                    if current_station in train_route['Station Code'].values:
                        path.append(current_station)


                ordered_stations_in_route = train_route[train_route['Station Code'].isin(stations_list)].sort_values('SEQ')
                for i in range(len(ordered_stations_in_route) -1 ) :
                    station1_data = ordered_stations_in_route.iloc[i]
                    station2_data = ordered_stations_in_route.iloc[i+1]
                    distance = abs(float(station2_data['Distance']) - float(station1_data['Distance']))
                    total_distance += distance


                if start_station in path and end_station in path:
                      return path, total_distance

        print("No single train solution found. Constructing path using multiple trains/transfers.")
        total_distance = 0
        path = []
        current_station = start_station

        for next_station in stations_list[1:]:  # Iterate through stations after start
            # Find a train connecting current_station and next_station
            connecting_train = None
            for train in df['Train No'].unique():
                train_route = df[df['Train No'] == train].sort_values('SEQ')
                if current_station in train_route['Station Code'].values and next_station in train_route['Station Code'].values:
                    connecting_train = train
                    break

            if connecting_train:
                # ... (Extract segment and calculate distance - same as before) ...
                current_station = next_station #Update current station

            else:  # No direct connection, find transfer station
                print(f"No direct train between {current_station} and {next_station}. Finding transfer...")
                transfer_station = find_transfer_station(df, current_station, next_station)

                if transfer_station:
                    path1, dist1 = get_segment_path_distance(df, current_station, transfer_station)
                    path2, dist2 = get_segment_path_distance(df, transfer_station, next_station)


                    if path1 and path2:
                        path.extend(path1)
                        path.extend(path2[1:])  #Avoid duplicate entries
                        total_distance += (dist1 + dist2)
                        current_station = next_station
                    else:
                        print(f"No transfer route found between {current_station} and {next_station}.")

                        return None, None  # Handle this case (skip, etc.)
                else:
                    print(f"No suitable transfer station found between {current_station} and {next_station}.")
                    return None, None  # Or handle differently



        return path, total_distance

    except Exception as e:
        print(f"Error finding original path: {e}")
        return None, None

        return path, total_distance #Return the combined path

    except Exception as e:
        print(f"Error finding original path: {e}")
        return None, None
def find_transfer_station(df, station1, station2):
    """Finds a common station to transfer between for station1 and station2."""
    trains1 = set(df[df['Station Code'] == station1]['Train No'].unique())
    trains2 = set(df[df['Station Code'] == station2]['Train No'].unique())

    # Stations reachable from station1
    reachable_from_s1 = set(df[df['Train No'].isin(trains1)]['Station Code'].unique())
    # Stations reachable from station2
    reachable_from_s2 = set(df[df['Train No'].isin(trains2)]['Station Code'].unique())
    
    # Find common reachable stations (potential transfer points)
    transfer_stations = reachable_from_s1.intersection(reachable_from_s2)
    if transfer_stations:
        return random.choice(list(transfer_stations))  # Or choose based on other criteria
    else:
        return None

def get_segment_path_distance(df, station1, station2):
    for train in df['Train No'].unique():
        train_route = df[df['Train No'] == train].sort_values('SEQ')
        if station1 in train_route['Station Code'].values and station2 in train_route['Station Code'].values:

            segment = train_route[train_route['Station Code'].isin([station1, station2])]


            start_idx = segment[segment['Station Code'] == station1].index[0]
            end_idx = segment[segment['Station Code'] == station2].index[0]
            path = segment.loc[min(start_idx, end_idx): max(start_idx, end_idx), 'Station Code'].tolist()
            distance = abs(float(segment[segment['Station Code'] == station1]['Distance']) - float(segment[segment['Station Code'] == station2]['Distance']))
            return path, distance

    return None, None
def plot_optimized_network(subgraph, stations_list, optimized_path, start_station, end_station):
    """
    Plot the subgraph with the optimized path highlighted
    """
    try:
        plt.figure(figsize=(12, 8))

        # Create layout for graph visualization
        pos = nx.spring_layout(subgraph, k=1, iterations=50)

        # Draw the network
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray')
        nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color='lightblue')

        # Draw station labels
        labels = {station: station for station in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

        # Highlight optimized path
        path_edges = list(zip(optimized_path[:-1], optimized_path[1:]))
        path_edges = [(stations_list[i], stations_list[j]) for i, j in path_edges]

        nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, 
                             edge_color='r', width=2)

        # Highlight start and end stations
        start_pos = {start_station: pos[start_station]}
        end_pos = {end_station: pos[end_station]}
        nx.draw_networkx_nodes(subgraph, start_pos, node_color='g', node_size=200)
        nx.draw_networkx_nodes(subgraph, end_pos, node_color='r', node_size=200)

        plt.title('Railway Network with Optimized Path')
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error plotting network: {e}")

def main():
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data('Train_details_22122017.csv')
    if df is None:
        return
    # Prepare features for demand forecasting
    print("Preparing features...")
    features = prepare_features_for_forecasting(df)
    if features is None:
        return
    
    target = features['train_count']  # Using number of trains as demand
    
    # Train demand forecasting model
    print("Training model...")
    model, scaler, mae, X_test, y_test, y_pred = train_demand_model(features, target)
    if model is None:
        return
    
    print(f"\nDemand Forecasting Results:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print("\nSample Predictions vs Actual:")
    for i in range(5):
        print(f"Actual: {y_test.iloc[i]:.0f}, Predicted: {y_pred[i]:.0f}")
    
    # Select a sample route
    sample_train = df['Train No'].iloc[0]
    sample_route = df[df['Train No'] == sample_train]
    start_station = sample_route.iloc[0]['Station Code']
    end_station = sample_route.iloc[-1]['Station Code']

    # Extract connected subgraph
    print("Extracting subgraph...")
    stations_list, subgraph = extract_connected_subgraph(
        df, start_station, end_station, num_nodes=20
    )

    if stations_list is None or subgraph is None:
        return

    # Create distance matrix from subgraph
    print("Creating distance matrix...")
    distance_matrix = create_distance_matrix_from_subgraph(subgraph, stations_list)
    if distance_matrix is None:
        return

    # Get start and end indices in the subgraph
    start_idx = stations_list.index(start_station)
    end_idx = stations_list.index(end_station)
    current_path, current_distance = get_original_path(df, start_station, end_station,stations_list)
    if current_path:
        print("Current Path:")
        print(" -> ".join(current_path))
        print(f"Current Distance: {current_distance:.2f} km")
    # Optimize path
    print("Finding optimal path...")
    optimized_path = optimize_path(distance_matrix, start_idx, end_idx)

    if optimized_path:
        print("\nPath Optimization Results:")
        print("Optimized Path:")
        path_stations = [stations_list[i] for i in optimized_path]
        print(" -> ".join(path_stations))

        # Calculate total distance
        total_distance = sum(distance_matrix[optimized_path[i]][optimized_path[i+1]] 
                           for i in range(len(optimized_path)-1))
        print(f"\nTotal Distance: {total_distance:.2f} km")

        # Visualize the network and path
        plot_optimized_network(subgraph, stations_list, optimized_path, 
                             start_station, end_station)

if __name__ == "__main__":
    main()