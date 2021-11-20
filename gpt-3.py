"""
Using the OpenAI API for GPT-3 https://beta.openai.com/
"""
import os
import itertools
from typing import List

import openai

openai.api_key = os.getenv('OPENAI_API_KEY')


def test_gpt3():
    prompt = "This is a question answerer.\n\n Question: What is your name?\nAnswer:"

    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n\n"]
    )

    print(response)


def chatbot():
    # Start the chatbot off
    prompt = 'This is a chatbot.\n\n'
    while True:
        # Get next question from the user
        question = input('You > ')
        # Ask GPT-3 for its answer
        prompt += f"Question: {question}\n  Answer:"
        response = openai.Completion.create(
            engine='davinci',
            prompt=prompt,
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        # Show answer to the user, and add to the prompt for next time
        answer = response['choices'][0]['text']
        print('GPT-3 >' + answer)
        prompt += answer + "\n"


def generate_names(num_names: int, place_type: str, example_fake_places: List[str], real_places: List[str]):
    # Apply a stricter "real places" constraint to GPT-3: No part of any real place can appear in GPT-3's places
    #   ignoring some common words (pre-words, post-words, connectives, etc.)
    real_places = set(itertools.chain.from_iterable(place.split(' ') for place in real_places))
    real_places.difference_update(['St.', 'St', 'Little', 'Greater', 'Great', 'Over', 'Under', 'On', 'Hill', 'Cross',
                                   'Cliffs', 'Cliff', 'Tor', 'Mere', 'Beck', 'Ghyll', 'Grange', 'The', 'Short', 'Small',
                                   'Old', 'End', 'Mill', 'North', 'South', 'East', 'West'])

    def is_real_place(possible_place: str) -> bool:
        parts = set(possible_place.split(' '))
        return len(parts.intersection(real_places)) > 0

    # Set up prompt showing GPT-3 what to produce
    example_places_text = '\n'.join('* ' + place for place in example_fake_places)
    prompt = f"This is a generator of fake {place_type} places, to trick humans.\n{example_places_text}\n*"

    print(prompt)
    print('------------')

    fake_places = []
    while len(fake_places) < num_names:

        # Run GPT-3
        response = openai.Completion.create(
            engine='davinci',
            prompt=prompt,
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        gpt3_response = response['choices'][0]['text'].strip()
        suggested_places = gpt3_response.split("\n* ")
        suggested_places = [place.replace("\n*", '').strip() for place in suggested_places]

        # Accumulate fake, unique names from GPT3's output, until we have enough
        new_places = list(set([place for place in suggested_places if not is_real_place(place) and place not in fake_places]))
        if len(new_places) + len(fake_places) > num_names:
            # Got enough, so just add the number requested
            places_to_add = new_places[:(num_names - len(fake_places))]
        else:
            # Not got enough, so add them all and run GPT-3 again
            places_to_add = new_places
        fake_places.extend(places_to_add)
        print('{num_done}/{total} ({percent}%): {new_ones}'.format(num_done=len(fake_places), total=num_names,
                                                                   percent=100.0*len(fake_places)/num_names, new_ones=places_to_add))

    return fake_places


def read_real_places_file(real_places_filename):
    data = open(real_places_filename, 'r').read()  # should be simple plain text file
    real_places = data.split("\n")
    real_places = list(set(real_places))
    return real_places


def generate_fake_places_to_file(num_names: int, place_type: str, example_fake_places: List[str],
                                 real_places_filename: str, fake_places_filename: str, variable_name: str):
    names = generate_names(num_names=num_names, place_type=place_type, example_fake_places=example_fake_places,
                           real_places=read_real_places_file(real_places_filename))
    with open(fake_places_filename, 'w') as file:
        file.write('{0} = [\n'.format(variable_name))
        for name in names:
            file.write("'{0}', \n".format(name.replace("'", "\\'")))
        file.write(']\n')


if __name__ == '__main__':
    # test_gpt3()
    # chatbot()
    # generate_names(num_names=50, place_type='English',
    #                example_fake_places=['Hillan', 'Little Peigraver', 'Holtrow Cross', 'Dunderdale', 'Pygill', 'Skidderdale', 'Little Ordham'],
    #                real_places=read_real_places_file('english-places.txt'))
    generate_fake_places_to_file(
        num_names=10_000, place_type='English',
        example_fake_places=['Hillan', 'Little Peigraver', 'Holtrow Cross', 'Dunderdale', 'Pygill', 'Skidderdale', 'Little Ordham'],
        real_places_filename='english-places.txt', fake_places_filename='gpt3_fake_english_places.js', variable_name='gpt3_english_places')
