import cupy as cp


def mean_back(gradient, op1):
    return (gradient / op1.size,)


def matmul_back(gradient, op1, op2):
    return (gradient @ op2.T, op1.T @ gradient)


class Node:
    def __init__(self, data, child1=None, child2=None, fn=None, leaf=True):
        self.data = data
        self.grad = cp.zeros_like(data)
        self.child1 = child1
        self.child2 = child2
        self.fn = fn
        self.leaf = leaf

    def backward(self, grad=None):
        nodes = [self]
        while nodes:
            cur_node = nodes.pop()
            if not cur_node.leaf:
                nodes.append(cur_node.child1)
                nodes.append(cur_node.child2) if cur_node.child2 else ...
            # create lazy eval graph
            print(cur_node.data, "\n") 

    def mean(self):
        return Node(
            data=self.data.mean(),
            child1=self,
            child2=None,
            fn=mean_back,
            leaf=False
        )

    def __matmul__(self, other):
        return Node(
            data=self.data @ other.data,
            child1=self,
            child2=other,
            fn=matmul_back,
            leaf=False
        )


def main():
    A = Node(cp.random.randn(2, 1))
    B = Node(cp.random.randn(1, 2))
    
    C = Node(cp.random.randn(2, 1))
    D = Node(cp.random.randn(1, 2))
    
    E = (A @ B) @ (C @ D)
    E.backward()


if __name__ == "__main__":
    main()
