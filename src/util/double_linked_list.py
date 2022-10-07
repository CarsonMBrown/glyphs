class DoubleLinkedNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def add_left(self, val):
        self.left = DoubleLinkedNode(val)
        self.left.right = self
        return self.left

    def add_right(self, val):
        self.right = DoubleLinkedNode(val)
        self.right.left = self
        return self.right

    def pop_left(self):
        temp = self.left
        self.left.right = None
        self.left = None
        return temp

    def pop_right(self):
        temp = self.right
        self.right.left = None
        self.right = None
        return temp

    def __str__(self):
        return str(self.val)

    def __eq__(self, other):
        if self == other or self.val == other:
            return True
