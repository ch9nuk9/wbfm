
def xml_pretty_print(root, max_depth=1, current_depth=1):
    for child in root:
        print(child.tag)
        print(child.attrib)
        if current_depth < max_depth:
            xml_pretty_print(child, max_depth=max_depth, current_depth=current_depth+1)

            
def dict_pretty_print(d, indent=0, max_indent=1000):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            if indent + 1 < max_indent:
                dict_pretty_print(value, indent+1)
            else:
                dict_pretty_print({'...':''}, indent+1)
        else:
            print('\t' * (indent+1) + str(value))