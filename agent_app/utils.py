def render_grid(env):
    """
    Renders the grid environment as an ASCII string.
    """
    grid_str = ""
    for r in range(env.height):
        row_str = ""
        for c in range(env.width):
            if (r, c) == env.agent_pos:
                row_str += "A "
            elif (r, c) == env.goal_pos:
                row_str += "G "
            elif env.grid[r, c] == 1:
                row_str += "# "
            elif (r, c) in env.visited and (r, c) != env.start_pos:
                 row_str += ". "
            else:
                row_str += "_ " # Using underscore for empty visibility
        grid_str += row_str + "\n"
    return grid_str
