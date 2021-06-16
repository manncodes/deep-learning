from pathlib import Path

# edit the below array to add directories and folders you want to exclude
ignore = [".git", ".ipynb_checkpoints", "test", "dataset", "runs"]


class DisplayablePath(object):
    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        self.depth = self.parent.depth + 1 if self.parent else 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            [path for path in root.iterdir() if criteria(path)],
            key=lambda s: str(s).lower(),
        )
        for count, path in enumerate(children, start=1):
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path, parent=displayable_root, is_last=is_last, criteria=criteria
                )
            else:
                yield cls(path, displayable_root, is_last)

    @classmethod
    def _default_criteria(cls, path):  # sourcery skip
        for dir in ignore:
            if str(path).find(dir) != -1:
                return False
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))


paths = DisplayablePath.make_tree(Path("."))
tree = [path.displayable() for path in paths]

with open("README.md", "w+", encoding="utf-8") as f:  # r+ does the work of rw
    f.write("# My Journey to learn Deep Learning\n")
    f.write("\n")
    f.write("## Directory Structure\n")
    f.write("\n")
    f.write("```bash")
    f.write("\n")
    for line in tree:
        f.write(line + "\n")
    f.write("```")
