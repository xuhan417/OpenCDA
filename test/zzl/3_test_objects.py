from itertools import groupby

def make_an_object(vehicles_list, tl_list):
    objects = {"vehicles": [], "traffic_lights": []}

    objects.update({"vehicles": vehicles_list})
    objects.update({"traffic_lights": tl_list})

    return objects


def update_dict_list(objects_all, objects, key):
    item_list = objects_all[key]
    item_list_cp = objects[key]
    item_list.extend(item_list_cp)
    objects_all.update({key: item_list})
    return objects_all


if __name__ == "__main__":
    objects_v2v = {"vehicles": [], "traffic_lights": []}

    vehicles_list1 = [[1, 1, 0], [3, 3, 0]]
    tl_list1 = [[-5, -1, 0], [10, 5, 0]]

    objects1 = make_an_object(vehicles_list1, tl_list1)

    vehicles_list2 = [[3, 1, 0], [3, 3, 0], [3, 5, 0]]
    tl_list2 = [[10, 5, 0], [5, 5, 4]]

    objects2 = make_an_object(vehicles_list2, tl_list2)

    objects_v2v = update_dict_list(objects_v2v, objects1, "vehicles")
    objects_v2v = update_dict_list(objects_v2v, objects2, "vehicles")

    """
    vehicles_list = []
    vehicles_list1_cp = objects1['vehicles']
    vehicles_list.extend(vehicles_list1_cp)
    
    vehicles_list2_cp = objects2['vehicles']
    vehicles_list.extend(vehicles_list2_cp)
    
    objects_v2v.update({'vehicles':vehicles_list})
    """

    # This will overwrite the previous value
    # objects_v2v.update(objects2)

    # objects_v2v.update({'vehicles':objects2['vehicles']})

    # This method will put all vehicles_lists together
    """
    vehicles_list = []
    vehicles_list.extend(vehicles_list1)
    vehicles_list.extend(vehicles_list2)
    """
