import torch


def load_data():
    class_names = ["Father", "Mother", "Parent", "Child", "Male", "Female"]
    individual_names = ["Mark", "Alice"]
    relation_names = ["hasChild", "hasParent"]

    classes = {c: i for i, c in enumerate(class_names)}
    relations = {r: i for i, r in enumerate(relation_names)}
    individuals = {i: j for j, i in enumerate(individual_names)}

    nf1 = [
        ("Father", "Male"),
        ("Mother", "Female"),
        ("Mother", "Parent"),
        ("Father", "Parent"),
    ]
    nf2 = [
        ("Male", "Parent", "Father"),
        ("Female", "Parent", "Mother")
    ]
    nf3 = [
        ("Parent", "hasChild", "Child"),
        ("Child", "hasParent", "Mother"),
        ("Child", "hasParent", "Father"),
    ]
    disjoint = [
        ("Male", "Female"),
        ("Parent", "Child"),
        ("Father", "Mother"),
    ]
    concept_assertions = [
        # ("Mark", "Father"),
        # ("Alice", "Mother"),
        # ("Mark", "Parent"),
        # ("Alice", "Parent"),
        # ("Mark", "Male"),
        # ("Alice", "Female"),
        ]

    data = {
        "nf1": to_tensor(nf1, classes, relations),
        "nf2": to_tensor(nf2, classes, relations),
        "nf3": to_tensor(nf3, classes, relations, use_relations=True),
        "nf4": [],
        "nf3_neg0": [],
        "disjoint": to_tensor(disjoint, classes, relations),
        # "abox": {
        #     "role_assertions": [],
        #     "concept_assertions": to_tensor(concept_assertions, classes, relations, individuals=individuals, use_individuals=True),
        #     "role_assertions_neg": []
        # }
    }
    return data, classes, relations, individuals


def to_tensor(data, classes, relations, individuals=None, use_relations=False,
              use_individuals=False):
    if use_relations:
        return torch.tensor(
            [[classes[tup[0]], relations[tup[1]], classes[tup[2]]] for tup in data],
            dtype=torch.long,
        )
    if use_individuals:
        return torch.tensor(
            [[classes[tup[1]], individuals[tup[0]]] for tup in data],
            dtype=torch.long,
        )
    return torch.tensor([list(map(classes.get, tup)) for tup in data], dtype=torch.long)
