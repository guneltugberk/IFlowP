import streamlit as st

def main():
    about = st.Page(
        page='directory/Home.py',
        title='About Project',
        icon='❓',
        default=True
    )

    extraction = st.Page(
        page='directory/First_Back_Pressure.py',
        title='First Back Pressure',
        icon='⛑️'
    )

    prediction = st.Page(
        page='directory/Second_Back_Pressure.py',
        title='Second Back Pressure',
        icon='💡'
    )

    pg = st.navigation(
        {
            'Info': [about],
            'Projects': [extraction, prediction]
        }
    )

    pg.run()

if __name__ == '__main__':
    main()


    
    