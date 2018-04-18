import os

import cv2
from img_utils.files import images_in_dir, filename

from l_vgg_face.feature import extract_feature, cos_distance
from l_vgg_face.front_face_extractor import extract
from l_vgg_face.known import KnownPerson


def read_from_hierarchy(images_dir):
    sub_dirs = [x for x in os.walk(images_dir)]
    hierarchy = {}
    for d in sub_dirs[0][1]:
        hierarchy[d] = os.path.join(images_dir, d)
    print(hierarchy)
    return hierarchy


def read_known_persons_from_hierarchical_images(images_dir):
    im_hierarchy = read_from_hierarchy(images_dir)

    known_person = KnownPerson()

    for k in im_hierarchy.keys():
        im_dir = im_hierarchy[k]
        im_files = images_in_dir(images_dir=im_dir)
        features = []
        for im_f in im_files:
            face = extract(im_f)
            features.append(extract_feature(face))

        known_person.add_person(k, features)
        known_person.save('known_person.pkl')
    return known_person


def read_known_persons(known_pickle=None, known_persons_hierarchical_images_dir=None):
    assert known_pickle is not None or known_persons_hierarchical_images_dir is not None
    if known_pickle is not None:
        known_person = KnownPerson.load(known_pickle)
    else:
        known_person = read_known_persons_from_hierarchical_images(known_persons_hierarchical_images_dir)
    return known_person


known_person_path = os.path.join(os.path.dirname(__file__), 'known_person.pkl')

# known = read_known_persons(known_persons_hierarchical_images_dir='/Users/administrator/Downloads/wg-colleagues/test')
known = read_known_persons(known_persons_hierarchical_images_dir='/Users/administrator/Downloads/wg-colleagues/labeled')
# known = read_known_persons('known_person.pkl')


def find_top_n_closest(face, n=3):
    feature = extract_feature(face)
    # cv2.imshow('feature', feature)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    results = []
    for k in known.known_list:
        for f in k['features']:
            dist = cos_distance(feature, f)
            results.append((k['name'], dist))
    results = sorted(results, key=lambda r: r[1])
    print(results)
    n = min(len(results), n)
    return results[0:n]


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def overlay(im, results):
    for i, r in enumerate(results):
        txt = '{}: {}'.format(r[0], r[1])
        cv2.putText(im, txt, (30, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return im


def predict(images_dir):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_files = images_in_dir(images_dir)
    for im_f in image_files:
        f_name = filename(im_f)
        im = cv2.imread(im_f)
        face = extract(im_f)
        results = find_top_n_closest(face)
        im = overlay(im, results)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f_name), im)


print(sorted([x['name'] for x in known.known_list]))
predict('/Users/administrator/Downloads/wg-colleagues/unlabled')
# predict('/Users/administrator/Downloads/wg-colleagues/xxx')
