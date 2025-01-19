import re
from Osu.OsuData import OsuData
from Osu.OsuUser import OsuUser


osudata = OsuData()
while True:
    user_id = input("Введите ID пользователя Osu (или 'выход' для завершения): ")
    user_id = re.search(r'users/(\d+)', user_id).group(1)
    if user_id.lower() == 'выход':
        break
    try:
        user = OsuUser(int(user_id))
        user.save()
        print(f"Пользователь с ID {user_id} сохранён.")
    except ValueError:
        print("Некорректный ввод, попробуйте снова.",ValueError)