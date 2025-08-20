def retrieve_known():
    # get all known action verbs
    with open("actionverbs.txt", 'r') as av:
        avlist = av.readlines()
    # get known action maps
    AM = {}
    with open("actionmaps.txt", 'r') as am:
        lines = am.readlines()
        for line in lines:
            line = line.strip()
            verb, steps = line.split(":", 1)
            step_list = [step.strip() for step in steps.split(", ")]
            AM[verb] = step_list
    # get known object attributes
    OB = {}
    with open("qualities.txt", 'r') as am:
        lines = am.readlines()
        for line in lines:
            line = line.strip()
            key, qualities = line.split(":", 1)
            quality_list = [qualities.strip() for quality in qualities.split(", ")]
            OB[key] = quality_list
    # get all known objects
    with open("objects.txt", 'r') as ob:
        objects = ob.readlines()
    return AM, avlist, OB, objects


action_map, action_verb_list, quality_map, object_list = retrieve_known()


def rebuild_instructions(initial_instruction, action_map, action_verb_list, quality_map, objects):
    instruction = initial_instruction.split()
    first_step = []
    # first identify action words, identifying words.
    # Example: click on the green apple → ACTION click COLOR green OBJECT apple
    for token in instruction:
        string = ""
        if token in action_verb_list:
            string = "ACTION "+token
        elif token in objects:
            string = "OBJECT "+token
        elif token in quality_map:
            string = quality_map[token] +" "+token
        if string:
            first_step.append(string)


    # second Rearrange so it reads like "ACTION OBJECT QUALITY"
    # Example: ACTION click on COLOR green OBJECT apple → ACTION click OBJECT apple COLOR green

    action_tokens = [t for t in first_step if t.startswith("ACTION")]
    object_tokens = [t for t in first_step if t.startswith("OBJECT")]
    quality_tokens = [t for t in first_step if not t.startswith("ACTION") and not t.startswith("OBJECT")]

    second_step = action_tokens + object_tokens + quality_tokens

    # third use computer vision to locate the object then verify the qualities to get the coordinates for the next click

    # fourth replace the object and qualities with the coordinates (x,y)


def update(word):
    file = ""
    word_type = int(input(f"Is {word} a action verb, object, quality; eg. 1, 2, 3\n"))
    match word_type:
        case 1:
            file = "actionverbs.txt"
        case 2:
            file = "objects.txt"
        case 3:
            file = "qualities.txt"

