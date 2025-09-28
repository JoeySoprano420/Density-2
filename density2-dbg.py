def start_debugger(ast):
    stack = [(ast, 0)]
    while stack:
        node, depth = stack.pop()
        print(f"[{depth}] Node: {node}")
        print(f"[Glyph] {node.to_dodecagram()}")
        cmd = input("> ").strip()
        if cmd == "print":
            print(node)
        elif cmd == "glyph":
            print(node.to_dodecagram())
        elif cmd == "history":
            for m in node.mutations:
                print(f"Mutation #{m.id}: {m.description}")
        elif cmd == "next":
            children = node.get_children()
            for child in reversed(children):
                stack.append((child, depth + 1))
        elif cmd == "quit":
            break

