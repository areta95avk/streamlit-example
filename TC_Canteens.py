import streamlit as st
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

st.title("Центр Торговли. Москва")
st.write("(быстрый прототип решения)")

## You can access the value at any point with:  # ЗАГОТОВКА на будущее
#st.session_state.name
#abc = st.session_state.name

# Раскладка продуктов на 2 порции 
meal_requirements = {
    "ДОМАШНИЕ ЕЖИКИ В ТОМАТНО-СМЕТАННОМ СОУСЕ С БУЛГУРОМ И ПТИТИМОМ": {"Птитим":150,"Томатная паста":50,"Фарш домашний":300,"Булгур":30,"Яйцо куриное":1,"Лук репчатый":1,"Чеснок":1,"Молоко":100,"Сметана 20%":5,"Сахар":5},
    "АДЖАПСАНДАЛИ - Грузинское овощное рагу с индейкой и хмели-сунели": {"Филе индейки (кусочки)":300,"Томаты протертые":100,"Хмели-сунели":2,"Баклажаны":1,"Перец":1,"Лук красный":1,"Помидоры":1,"Морковь":1,"Петрушка":2,"Кинза":2,"Чеснок":2,"Растительное масло":1,"Соль, перец - по вкусу":1},
    "ПИЦЦА МАРИНАРА с МОЦАРЕЛЛОЙ И ОЛИВКАМИ":{'Тесто для пиццы': 300,'Соус томатный для пиццы':140,'Сыр Моцарелла тертый':100,'Орегано': 1,'Чеснок':2,'Оливки':30,'Оливковое масло':20,'Мука пшеничная':10,'Пергамент':1},
    "Спагетти Болоньезе с говядиной по-мексикански": {"Спагетти": 200,"Фарш говяжий": 300,"Томаты вяленые":20,"Фасоль красная вареная":100,"Чеснок": 3,"Кинза": 2,"Перец Чили":1,"Лук красный":1,"Томаты протертые":150},
    "Шницель из индейки с отварным картофелем и ароматным зеленым маслом":{"Филе индейки (цельное)":300,"Панировочные сухари":80,"Яйцо куриное":1,"Картофель":550,"Сливочное масло":30,"Петрушка":2,"Укроп":2,"Чеснок":2,"Лимон":1}
    }

if 'df_dishes_main' not in st.session_state:
    st.session_state.df_dishes_main = pd.DataFrame(meal_requirements ).fillna(0).astype('int64')
    st.session_state.df_dishes_main["Stock_limit"] = [0]* 41
    st.session_state.df_dishes_main["Stock_limit"] = [0]* 41
    st.session_state.df_dishes_main['Stock_limit'] = st.session_state.df_dishes_main.iloc[:,0:5].sum(axis=1) * 1000
    st.session_state.df_dishes_main['Stock'] = st.session_state.df_dishes_main['Stock_limit'] - 50 * st.session_state.df_dishes_main.iloc[:,0] - 50 * st.session_state.df_dishes_main.iloc[:,1] 

if st.checkbox('Посмотреть и уточнить запасы продуктов'):
    df_dishes = st.session_state.df_dishes_main
    df_dishes.iloc[0,6] = st.slider(df_dishes.index[0], 0, df_dishes.iloc[0,5], df_dishes.iloc[0,6])
    df_dishes.iloc[1,6] = st.slider(df_dishes.index[1], 0, df_dishes.iloc[1,5], df_dishes.iloc[1,6])
    df_dishes.iloc[2,6] = st.slider(df_dishes.index[2], 0, df_dishes.iloc[2,5], df_dishes.iloc[2,6]) 
    df_dishes.iloc[3,6] = st.slider(df_dishes.index[3], 0, df_dishes.iloc[3,5], df_dishes.iloc[3,6]) 
    df_dishes.iloc[4,6] = st.slider(df_dishes.index[4], 0, df_dishes.iloc[4,5], df_dishes.iloc[4,6])
    df_dishes.iloc[5,6] = st.slider(df_dishes.index[5], 0, df_dishes.iloc[5,5], df_dishes.iloc[5,6])
    df_dishes.iloc[6,6] = st.slider(df_dishes.index[6], 0, df_dishes.iloc[6,5], df_dishes.iloc[6,6])
    df_dishes.iloc[7,6] = st.slider(df_dishes.index[7], 0, df_dishes.iloc[7,5], df_dishes.iloc[7,6])
    df_dishes.iloc[8,6] = st.slider(df_dishes.index[8], 0, df_dishes.iloc[8,5], df_dishes.iloc[8,6])
    df_dishes.iloc[9,6] = st.slider(df_dishes.index[9], 0, df_dishes.iloc[9,5], df_dishes.iloc[9,6]) 
    df_dishes.iloc[10,6] = st.slider(df_dishes.index[10], 0, df_dishes.iloc[10,5], df_dishes.iloc[10,6])
    df_dishes.iloc[11,6] = st.slider(df_dishes.index[11], 0, df_dishes.iloc[11,5], df_dishes.iloc[11,6])
    df_dishes.iloc[12,6] = st.slider(df_dishes.index[12], 0, df_dishes.iloc[12,5], df_dishes.iloc[12,6])
    df_dishes.iloc[13,6] = st.slider(df_dishes.index[13], 0, df_dishes.iloc[13,5], df_dishes.iloc[13,6])
    df_dishes.iloc[14,6] = st.slider(df_dishes.index[14], 0, df_dishes.iloc[14,5], df_dishes.iloc[14,6])
    df_dishes.iloc[15,6] = st.slider(df_dishes.index[15], 0, df_dishes.iloc[15,5], df_dishes.iloc[15,6]) 
    df_dishes.iloc[16,6] = st.slider(df_dishes.index[16], 0, df_dishes.iloc[16,5], df_dishes.iloc[16,6])
    df_dishes.iloc[17,6] = st.slider(df_dishes.index[17], 0, df_dishes.iloc[17,5], df_dishes.iloc[17,6])
    df_dishes.iloc[18,6] = st.slider(df_dishes.index[18], 0, df_dishes.iloc[18,5], df_dishes.iloc[18,6])
    df_dishes.iloc[19,6] = st.slider(df_dishes.index[19], 0, df_dishes.iloc[19,5], df_dishes.iloc[19,6])
    df_dishes.iloc[20,6] = st.slider(df_dishes.index[20], 0, df_dishes.iloc[20,5], df_dishes.iloc[20,6])
    df_dishes.iloc[21,6] = st.slider(df_dishes.index[21], 0, df_dishes.iloc[21,5], df_dishes.iloc[21,6]) 
    df_dishes.iloc[22,6] = st.slider(df_dishes.index[22], 0, df_dishes.iloc[22,5], df_dishes.iloc[22,6])
    df_dishes.iloc[23,6] = st.slider(df_dishes.index[23], 0, df_dishes.iloc[23,5], df_dishes.iloc[23,6])
    df_dishes.iloc[24,6] = st.slider(df_dishes.index[24], 0, df_dishes.iloc[24,5], df_dishes.iloc[24,6])
    df_dishes.iloc[25,6] = st.slider(df_dishes.index[25], 0, df_dishes.iloc[25,5], df_dishes.iloc[25,6])
    df_dishes.iloc[26,6] = st.slider(df_dishes.index[26], 0, df_dishes.iloc[26,5], df_dishes.iloc[26,6])
    df_dishes.iloc[27,6] = st.slider(df_dishes.index[27], 0, df_dishes.iloc[27,5], df_dishes.iloc[27,6]) 
    df_dishes.iloc[28,6] = st.slider(df_dishes.index[28], 0, df_dishes.iloc[28,5], df_dishes.iloc[28,6])
    df_dishes.iloc[29,6] = st.slider(df_dishes.index[29], 0, df_dishes.iloc[29,5], df_dishes.iloc[29,6])
    df_dishes.iloc[30,6] = st.slider(df_dishes.index[30], 0, df_dishes.iloc[30,5], df_dishes.iloc[30,6])
    df_dishes.iloc[31,6] = st.slider(df_dishes.index[31], 0, df_dishes.iloc[31,5], df_dishes.iloc[31,6])
    df_dishes.iloc[32,6] = st.slider(df_dishes.index[32], 0, df_dishes.iloc[32,5], df_dishes.iloc[32,6])
    df_dishes.iloc[33,6] = st.slider(df_dishes.index[33], 0, df_dishes.iloc[33,5], df_dishes.iloc[33,6]) 
    df_dishes.iloc[34,6] = st.slider(df_dishes.index[34], 0, df_dishes.iloc[34,5], df_dishes.iloc[34,6])
    df_dishes.iloc[35,6] = st.slider(df_dishes.index[35], 0, df_dishes.iloc[35,5], df_dishes.iloc[35,6])
    df_dishes.iloc[36,6] = st.slider(df_dishes.index[36], 0, df_dishes.iloc[36,5], df_dishes.iloc[36,6])
    df_dishes.iloc[37,6] = st.slider(df_dishes.index[37], 0, df_dishes.iloc[37,5], df_dishes.iloc[37,6])
    df_dishes.iloc[38,6] = st.slider(df_dishes.index[38], 0, df_dishes.iloc[38,5], df_dishes.iloc[38,6])
    df_dishes.iloc[39,6] = st.slider(df_dishes.index[39], 0, df_dishes.iloc[39,5], df_dishes.iloc[39,6]) 
    df_dishes.iloc[40,6] = st.slider(df_dishes.index[40], 0, df_dishes.iloc[40,5], df_dishes.iloc[40,6])
    
 # Кнопка для сохранения изменений
    if st.button('Сохранить изменения'):
            st.session_state.df_dishes_main = df_dishes
            st.success('Изменения сохранены!')
 



data = {
    'Min': [70,80,120,50,100],
    'Max': [250,500,500,500,400],
}

if 'df_order' not in st.session_state:
  st.session_state.df_order = pd.DataFrame(data,index=list(meal_requirements), columns = ['Min', 'Max'])



if st.checkbox('Сколько блюд готовить? Укажите нижний и верхний предел для каждого блюда.'):
    df_order2 = st.data_editor(st.session_state.df_order)
        # Кнопка для сохранения изменений
    if st.button('Сохранить изменения'):
            st.session_state.df_order = df_order2
            st.success('Изменения сохранены!')

 
total_portions = st.number_input("Какое количество гостей сегодня ожидается?", key="total_portions", value=800)
# You can access the value at any point with:
st.session_state.total_portions

if st.button('Выполнить расчет'):

    # Создаем модель CP
    model = cp_model.CpModel()

    meal1_name = list(meal_requirements)[0]
    meal2_name = list(meal_requirements)[1]
    meal3_name = list(meal_requirements)[2]
    meal4_name = list(meal_requirements)[3]
    meal5_name = list(meal_requirements)[4]

    # Переменные: количество порций каждого типа
    meal1 = model.NewIntVar(st.session_state.df_order.iloc[0,0], st.session_state.df_order.iloc[0,1], meal1_name)  
    meal2 = model.NewIntVar(st.session_state.df_order.iloc[1,0], st.session_state.df_order.iloc[1,1], meal2_name)       
    meal3 = model.NewIntVar(st.session_state.df_order.iloc[2,0], st.session_state.df_order.iloc[2,1], meal3_name)      
    meal4 = model.NewIntVar(st.session_state.df_order.iloc[3,0], st.session_state.df_order.iloc[3,1], meal4_name)      
    meal5 = model.NewIntVar(st.session_state.df_order.iloc[4,0], st.session_state.df_order.iloc[4,1], meal5_name)    

    meal1_d = model.NewIntVar(0, 500, 'meal1_d') 
    meal2_d = model.NewIntVar(0, 500, 'meal1_d')       
    meal3_d = model.NewIntVar(0, 500, 'meal1_d')      
    meal4_d = model.NewIntVar(0, 500, 'meal1_d')      
    meal5_d = model.NewIntVar(0, 500, 'meal1_d') 

    model.add_abs_equality(meal1_d, meal1-250)
    model.add_abs_equality(meal2_d, meal2-250)
    model.add_abs_equality(meal3_d, meal3-250)
    model.add_abs_equality(meal4_d, meal4-250)
    model.add_abs_equality(meal5_d, meal5-125)

    # Ограничение: общее количество порций = 100
    model.Add(meal1 + meal2 + meal3 + meal4 + meal5 == st.session_state.total_portions)

    # Ограничение: продукты не должны закончиться
    for product, stock in     dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock)).items():  
        total_used = 0
        for meal_type in list(meal_requirements):
            if product in meal_requirements[meal_type]:
                # Суммируем расход продукта по всем порциям
                total_used += meal_requirements[meal_type][product] * (
                    meal1 if meal_type == meal1_name 
                        else meal2 if meal_type == meal2_name 
                                    else meal3 if meal_type == meal3_name 
                                                else meal4 if meal_type == meal4_name 
                                                        else meal5
                )
        model.Add(total_used <= stock)  


    model.minimize(meal1_d + meal2_d + meal3_d + meal4_d + meal5_d )

    # Решаем
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Вывод результатов
    if status == cp_model.OPTIMAL:
        st.write(f"**Решение найдено:**")
        st.write(f"- {meal1_name}: {solver.Value(meal1)} порций")
        st.write(f"- {meal2_name}: {solver.Value(meal2)} порций")
        st.write(f"- {meal3_name}: {solver.Value(meal3)} порций")
        st.write(f"- {meal4_name}: {solver.Value(meal4)} порций")
        st.write(f"- {meal5_name}: {solver.Value(meal5)} порций")



# Проверка расхода продуктов
  
            # Остаток продуктов
        st.write(f"\n**Остаток продуктов:**")
        for product in dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock)):
            used = sum(
                meal_requirements[meal_type].get(product, 0) * solver.Value(
                    meal1 if meal_type == meal1_name else meal2 if meal_type == meal2_name else meal3 if meal_type == meal3_name 
                                else meal4 if meal_type == meal4_name  else meal5
                )
                for meal_type in [meal1_name, meal2_name, meal3_name,meal4_name,meal5_name ]
            )
            st.write(f"- {product}: использовано {int(used)} из {dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product]}, осталось {dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] - used}, {int((dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] - used) / dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] * 100)}%")
         #   st.write(f" {product}: вычесть {int(used)} из запасов {dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product]}")
         #   st.write(f" Stock {st.session_state.df_dishes_main.loc[product,'Stock']}")
         #   st.write(f" Что вычитаем {int(used)} ")
            st.session_state.df_dishes_main.loc[product,'Stock'] = st.session_state.df_dishes_main.loc[product,'Stock'] - int(used)
   
      #  if st.button('Списать продукты?'):
      #     for product in dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock)):
      #          used = sum(
      #              meal_requirements[meal_type].get(product, 0) * solver.Value(
      #                  meal1 if meal_type == meal1_name else meal2 if meal_type == meal2_name else meal3 if meal_type == meal3_name 
      #                          else meal4 if meal_type == meal4_name  else meal5
      #              )
      #          for meal_type in [meal1_name, meal2_name, meal3_name,meal4_name,meal5_name ]
      #          )
      #          st.write(f"- {product}: использовано {used:.1f} из {dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product]}, осталось {dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] - used}, {int((dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] - used) / dict(zip(st.session_state.df_dishes_main.index,st.session_state.df_dishes_main.Stock))[product] * 100)}%")
    else:
        st.write("Решение не найдено. Возможно, не хватает продуктов.")    