from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = None

    def movement_step(self, action: GridActions):
        old_position = self.state.position
        x, y = old_position

        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            self.landing_attempts += 1
            if self.state.is_in_landing_zone():
                self.state.set_landed(True)

        self.state.set_position([x, y])
        if self.state.is_in_no_fly_zone():
            # Reset state
            self.boundary_counter += 1
            x, y = old_position
            self.state.set_position([x, y])

        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        return x, y

    def reset(self, state):
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state
