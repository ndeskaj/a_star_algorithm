# A* algorithm with visualizer

The pseudocode:
```
function A_Star(start, goal):
    // Initialize open and closed lists
    openList = [start]          // Nodes to be evaluated
    closedList = []            // Nodes already evaluated
    
    // Initialize node properties
    start.g = 0                // Cost from start to start is 0
    start.h = heuristic(start, goal)  // Estimate to goal
    start.f = start.g + start.h       // Total estimated cost
    start.parent = null              // For path reconstruction
    while openList is not empty:
        // Get node with lowest f value - implement using a priority queue
       // for faster retrieval of the best node
        current = node in openList with lowest f value
        
        // Check if we've reached the goal
        if current = goal:
            return reconstruct_path(current)
            
        // Move current node from open to closed list
        remove current from openList
        add current to closedList
        
        // Check all neighboring nodes
        for each neighbor of current:
            if neighbor in closedList:
                continue  // Skip already evaluated nodes
                
            // Calculate tentative g score
            tentative_g = current.g + distance(current, neighbor)
            
            if neighbor not in openList:
                add neighbor to openList
            else if tentative_g >= neighbor.g:
                continue  // This path is not better
                
            // This path is the best so far
            neighbor.parent = current
            neighbor.g = tentative_g
            neighbor.h = heuristic(neighbor, goal)
            neighbor.f = neighbor.g + neighbor.h
    
    return failure  // No path exists
function reconstruct_path(current):
    path = []
    while current is not null:
        add current to beginning of path
        current = current.parent
    return path
```

(/images/screenshot.png)
