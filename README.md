## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
encoding_data = pd.read_csv("Encoding Data (1).csv")
data = pd.read_csv("data (1).csv")
transform_data = pd.read_csv("Data_to_Transform (1).csv")
print("ENCODING DATASET 1")
print(encoding_data.head())
print("\nENCODING DATASET 2")
print(data.head())
print("\nTRANSFORMATION DATASET")
print(transform_data.head())
def clean_data(df):

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove extra spaces from categorical values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Handle missing values
    for col in df.columns:

        if df[col].isnull().sum() > 0:

            if df[col].dtype == "object":
                df[col] = df[col].fillna(
                    df[col].mode()[0]
                )

            else:
                df[col] = df[col].fillna(
                    df[col].median()
                )

    return df


encoding_data = clean_data(encoding_data)
data = clean_data(data)
transform_data = clean_data(transform_data)

print("\nDATA CLEANING COMPLETED")


def label_encoding(df, columns):

    df = df.copy()

    for col in columns:

        categories = sorted(df[col].dropna().unique())

        mapping = {
            value: index
            for index, value in enumerate(categories)
        }

        df[col + "_Label"] = df[col].map(mapping)

        print("\nLabel Encoding:", col)
        print(mapping)

    return df


def ordinal_encoding(df, ordinal_maps):

    df = df.copy()

    for col, order in ordinal_maps.items():

        mapping = {
            value: index
            for index, value in enumerate(order)
        }

        unknown = set(df[col].dropna().unique()) - set(mapping)

        if unknown:
            raise ValueError(
                f"Unknown categories in {col}: {unknown}"
            )

        df[col + "_Ordinal"] = df[col].map(mapping)

        print("\nOrdinal Encoding:", col)
        print(mapping)

    return df


def binary_encoding(series, prefix):

    categories = sorted(
        series.dropna().unique()
    )

    mapping = {
        value: index
        for index, value in enumerate(categories)
    }

    codes = series.map(mapping).to_numpy(dtype=int)

    width = max(
        1,
        int(np.ceil(np.log2(max(len(categories), 2))))
    )

    result = pd.DataFrame(index=series.index)

    for bit in range(width):

        result[prefix + "_Bit_" + str(bit + 1)] = (
            (codes >> bit) & 1
        )

    print("\nBinary Encoding:", prefix)
    print(mapping)

    return result


def one_hot_encoding(df, columns):

    encoded = pd.get_dummies(
        df[columns],
        prefix=columns,
        dtype=int
    )

    return encoded



print("\n========== DATASET 1 ENCODING ==========")

encoded_data1 = label_encoding(
    encoding_data,
    ["bin_1", "bin_2"]
)

encoded_data1 = ordinal_encoding(
    encoded_data1,
    {
        "ord_2": ["Cold", "Warm", "Hot"]
    }
)

binary1 = binary_encoding(
    encoding_data["nom_0"],
    "nom_0"
)

encoded_data1 = pd.concat(
    [encoded_data1, binary1],
    axis=1
)

onehot1 = one_hot_encoding(
    encoding_data,
    ["nom_0"]
)

encoded_data1 = pd.concat(
    [encoded_data1, onehot1],
    axis=1
)

encoded_data1.to_csv(
    "Encoding_Data1_Output.csv",
    index=False
)

print("\nDATASET 1 FINAL ENCODED OUTPUT")
print(encoded_data1.head())


print("\n========== DATASET 2 ENCODING ==========")

encoded_data2 = label_encoding(
    data,
    ["bin_1", "bin_2"]
)

encoded_data2 = ordinal_encoding(
    encoded_data2,
    {
        "Ord_1": [
            "Cold",
            "Warm",
            "Hot",
            "Very Hot"
        ],

        "Ord_2": [
            "High School",
            "Diploma",
            "Bachelors",
            "Masters",
            "PhD"
        ]
    }
)

binary2 = binary_encoding(
    data["City"],
    "City"
)

encoded_data2 = pd.concat(
    [encoded_data2, binary2],
    axis=1
)

onehot2 = one_hot_encoding(
    data,
    ["City"]
)

encoded_data2 = pd.concat(
    [encoded_data2, onehot2],
    axis=1
)

encoded_data2.to_csv(
    "Encoding_Data2_Output.csv",
    index=False
)

print("\nDATASET 2 FINAL ENCODED OUTPUT")
print(encoded_data2.head())


print("\n========== FEATURE TRANSFORMATION ==========")

# Work with numerical columns
result = transform_data.copy()

numeric_columns = transform_data.select_dtypes(
    include=np.number
).columns



for col in numeric_columns:

    x = transform_data[col].astype(float)

    shift = max(0, 0.000001 - x.min())

    xp = x + shift

    # 1. LOG TRANSFORMATION

    result[col + "_Log"] = np.log(xp)

    # 2. RECIPROCAL TRANSFORMATION

    result[col + "_Reciprocal"] = 1 / xp

    # 3. SQUARE ROOT TRANSFORMATION

    result[col + "_SquareRoot"] = np.sqrt(xp)

    # 4. SQUARE TRANSFORMATION

    result[col + "_Square"] = x ** 2


# Display all four function transformation outputs

print("\n========== FUNCTION TRANSFORMATION OUTPUT ==========")

for col in numeric_columns:

    print("\nOriginal Feature:", col)

    print("\n1. Log Transformation")
    print(result[col + "_Log"].head())

    print("\n2. Reciprocal Transformation")
    print(result[col + "_Reciprocal"].head())

    print("\n3. Square Root Transformation")
    print(result[col + "_SquareRoot"].head())

    print("\n4. Square Transformation")
    print(result[col + "_Square"].head())


# Save function transformation output

function_columns = [
    col + suffix
    for col in numeric_columns
    for suffix in [
        "_Log",
        "_Reciprocal",
        "_SquareRoot",
        "_Square"
    ]
]

function_output = result[
    list(transform_data.columns) + function_columns
]

function_output.to_csv(
    "Function_Transformation_Output.csv",
    index=False
)


# ---------- POWER TRANSFORMATIONS ----------

print("\n========== POWER TRANSFORMATION OUTPUT ==========")

power_result = transform_data.copy()

for col in numeric_columns:

    x = transform_data[col].astype(float)

    # Box-Cox requires strictly positive values
    shift = max(0, 0.000001 - x.min())
    xp = x + shift

    # 5. BOX-COX TRANSFORMATION

    boxcox = PowerTransformer(
        method="box-cox",
        standardize=False
    )

    power_result[col + "_BoxCox"] = (
        boxcox.fit_transform(
            xp.to_numpy().reshape(-1, 1)
        ).ravel()
    )

    # 6. YEO-JOHNSON TRANSFORMATION

    yeojohnson = PowerTransformer(
        method="yeo-johnson",
        standardize=False
    )

    power_result[col + "_YeoJohnson"] = (
        yeojohnson.fit_transform(
            x.to_numpy().reshape(-1, 1)
        ).ravel()
    )

    print("\nFeature:", col)

    print("\n5. Box-Cox Transformation")
    print(power_result[col + "_BoxCox"].head())

    print("\n6. Yeo-Johnson Transformation")
    print(power_result[col + "_YeoJohnson"].head())


# Save power transformation output

power_columns = [
    col + suffix
    for col in numeric_columns
    for suffix in [
        "_BoxCox",
        "_YeoJohnson"
    ]
]

power_output = power_result[
    list(transform_data.columns) + power_columns
]

power_output.to_csv(
    "Power_Transformation_Output.csv",
    index=False
)


# ---------- SAVE ALL TRANSFORMATIONS TO ONE FILE ----------

result.to_csv(
    "All_Transformation_Output.csv",
    index=False
)


# =====================================================
# STEP 5: FINAL OUTPUT
# =====================================================

print("\n========== FINAL RESULTS ==========")

print("\nEncoded Dataset 1 Shape:")
print(encoded_data1.shape)

print("\nEncoded Dataset 2 Shape:")
print(encoded_data2.shape)

print("\nFunction Transformation Shape:")
print(function_output.shape)

print("\nPower Transformation Shape:")
print(power_output.shape)

print("\nAll Transformation Shape:")
print(result.shape)
```

# Output:
<img width="822" height="703" alt="Screenshot 2026-11-25 154554" src="https://github.com/user-attachments/assets/b5146e56-9210-4d7f-9c0d-c1c97f33f506" />
<img width="815" height="635" alt="Screenshot 2026-11-25 154615" src="https://github.com/user-attachments/assets/d0940e21-b801-4a40-9b5a-a598e46f478d" />
<img width="893" height="881" alt="Screenshot 2026-11-25 154744" src="https://github.com/user-attachments/assets/cf4af61a-c15c-4225-ac06-444d668f08a4" />
<img width="898" height="896" alt="Screenshot 2026-11-25 154839" src="https://github.com/user-attachments/assets/c658aa24-3733-43d9-bccd-b405fe841338" />
<img width="823" height="807" alt="Screenshot 2026-11-25 154911" src="https://github.com/user-attachments/assets/929f8eaf-e759-4d8d-b58b-a0dde551d4e6" />
<img width="821" height="663" alt="Screenshot 2026-11-25 154944" src="https://github.com/user-attachments/assets/960a3a53-6e3a-4111-aecf-1b78a1d7bff2" />
<img width="777" height="681" alt="Screenshot 2026-11-25 155001" src="https://github.com/user-attachments/assets/4a144d97-71c1-427f-9543-c6b0acdb70c7" />
<img width="822" height="727" alt="Screenshot 2026-11-25 155035" src="https://github.com/user-attachments/assets/2580841a-1acb-4124-a82a-8fd387d5e6ff" />
<img width="827" height="727" alt="Screenshot 2026-11-25 155104" src="https://github.com/user-attachments/assets/1f3342a6-b38c-46b2-8ebb-69e9621c3e73" />















 
       
# RESULT:
   successfully performed Feature Encoding and Transformation process
       
