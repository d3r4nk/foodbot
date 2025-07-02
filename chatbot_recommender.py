import pandas as pd
import re
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from collections import Counter, defaultdict

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChatbotRecommender:
    def __init__(self, df, similarity_matrices):
        self.df = df
        self.similarity_matrices = similarity_matrices
        self.indices = pd.Series(df.index, index=df['title'])
        
        self.model_weights = {   #cái nào có độ chính xác càng cao trong trang chính của food recommendation thì càng có trọng số cao 
            'doc2vec': 1.2,
            'tfidf': 1.0,
            'pyvi': 1.1,
            'bm25': 0.9,
            'vncorenlp': 1.3
        }
        
        self.vietnamese_stopwords = [
             # Đại từ
    'tôi', 'bạn', 'anh', 'chị', 'em', 'chúng tôi', 'chúng ta', 'họ', 'nó', 'mình',
    'ta', 'tụi tôi', 'chúng mình', 'ai', 'gì', 'đâu', 'sao', 'nào', 'nào đó',
    'kia', 'này', 'đó', 'ấy', 'đây', 'kìa', 'đấy', 'ngươi', 'người', 'tớ',
    'chú', 'cô', 'ông', 'bà', 'cậu', 'mợ', 'thím', 'ai ai', 'ai nấy', 'ai đó',
    'con', 'con dạ', 'con nhà', 'cu cậu', 'chú mày', 'chú mình', 'chúng ông',
    'cô mình', 'cô quả', 'cô tăng', 'cô ấy', 'anh ấy', 'chị ấy', 'ông ấy',
    'bà ấy', 'chị bộ', 'ông nhỏ', 'ông tạo', 'ông từ', 'ông ổng', 'chú dẫn',
    'chú khách',
    
    # Liên từ và từ nối
    'và', 'hay', 'hoặc', 'nhưng', 'mà', 'song', 'tuy', 'tuy nhiên', 'tuy vậy',
    'tuy thế', 'dù', 'dầu', 'cho dù', 'bởi vì', 'vì', 'do', 'nên', 'nên chi',
    'cho nên', 'vì vậy', 'vì thế', 'do đó', 'bởi thế', 'bởi vậy', 'thế nên',
    'vậy nên', 'thành ra', 'thành thử', 'vậy', 'thế', 'như vậy', 'như thế',
    'hoặc là', 'song le', 'nhiên hậu', 'thay đổi', 'thay đổi tình trạng',
    'dù cho', 'dù dì', 'dù gì', 'dù rằng', 'dù sao', 'dầu sao', 'dẫu',
    'dẫu mà', 'dẫu rằng', 'dẫu sao', 'nhưng mà', 'mà cả', 'mà không',
    'mà lại', 'mà thôi', 'mà vẫn',
    
    # Giới từ
    'của', 'cho', 'với', 'về', 'trong', 'ngoài', 'từ', 'đến', 'tới', 'qua',
    'theo', 'dọc', 'suốt', 'tại', 'ở', 'trên', 'dưới', 'giữa', 'sau', 'trước',
    'bên', 'cạnh', 'gần', 'xa', 'trong', 'ngoài', 'giữa', 'sang', 'lên', 'xuống',
    'vào', 'ra', 'về', 'đi', 'lại', 'qua lại', 'tới lui', 'bên bị', 'bên có',
    'bên cạnh', 'theo bước', 'theo như', 'theo tin', 'dọc theo', 'suốt đời',
    'tại lòng', 'tại nơi', 'tại tôi', 'tại đâu', 'tại đây', 'tại đó',
    'ở lại', 'ở như', 'ở nhờ', 'ở năm', 'ở trên', 'ở vào', 'ở đây', 'ở đó',
    'ở được', 'trên bộ', 'trên dưới', 'dưới nước', 'giữa lúc', 'sau chót',
    'sau cuối', 'sau cùng', 'sau hết', 'sau này', 'sau nữa', 'sau sau',
    'sau đây', 'sau đó', 'trước hết', 'trước khi', 'trước kia', 'trước nay',
    'trước ngày', 'trước nhất', 'trước sau', 'trước tiên', 'trước tuổi',
    'trước đây', 'trước đó', 'bên này', 'bên kia', 'bên ấy', 'cạnh bên',
    'gần bên', 'gần hết', 'gần ngày', 'gần như', 'gần xa', 'gần đây',
    'gần đến', 'xa cách', 'xa gần', 'xa nhà', 'xa tanh', 'xa tắp',
    'xa xa', 'xa xả', 'ngoài này', 'ngoài ra', 'ngoài xa', 'trong khi',
    'trong lúc', 'trong mình', 'trong ngoài', 'trong này', 'trong số',
    'trong vùng', 'trong đó', 'trong ấy', 'từ căn', 'từ giờ', 'từ khi',
    'từ loại', 'từ nay', 'từ thế', 'từ tính', 'từ tại', 'từ từ', 'từ ái',
    'từ điều', 'từ đó', 'từ ấy', 'đến bao giờ', 'đến cùng', 'đến cùng cực',
    'đến cả', 'đến giờ', 'đến gần', 'đến hay', 'đến khi', 'đến lúc',
    'đến lời', 'đến nay', 'đến ngày', 'đến nơi', 'đến nỗi', 'đến thì',
    'đến thế', 'đến tuổi', 'đến xem', 'đến điều', 'đến đâu', 'tới gần',
    'tới mức', 'tới nơi', 'tới thì', 'qua chuyện', 'qua khỏi', 'qua lại',
    'qua lần', 'qua ngày', 'qua tay', 'qua thì', 'qua đi', 'vào gặp',
    'vào khoảng', 'vào lúc', 'vào vùng', 'vào đến', 'ra bài', 'ra bộ',
    'ra chơi', 'ra gì', 'ra lại', 'ra lời', 'ra ngôi', 'ra người',
    'ra sao', 'ra tay', 'ra vào', 'ra ý', 'ra điều', 'ra đây',
    'về không', 'về nước', 'về phần', 'về sau', 'về tay', 'lên cao',
    'lên cơn', 'lên mạnh', 'lên ngôi', 'lên nước', 'lên số', 'lên xuống',
    'lên đến', 'xuống', 'xăm xúi', 'xăm xăm', 'xăm xắm', 'sang năm',
    'sang sáng', 'sang tay',
    
    # Động từ phụ trợ
    'là', 'thì', 'mà', 'sẽ', 'đã', 'đang', 'vẫn', 'còn', 'đều', 'cũng',
    'chỉ', 'chính', 'đúng', 'quả', 'thực', 'thật', 'hẳn', 'ắt', 'hẳn',
    'chắc', 'có lẽ', 'có thể', 'phải', 'cần', 'nên', 'được', 'bị',
    'sẽ biết', 'sẽ hay', 'đã hay', 'đã không', 'đã là', 'đã lâu',
    'đã thế', 'đã vậy', 'đã đủ', 'đang tay', 'đang thì', 'vẫn thế',
    'còn như', 'còn nữa', 'còn thời gian', 'còn về', 'đều bước',
    'đều nhau', 'đều đều', 'cũng như', 'cũng nên', 'cũng thế',
    'cũng vậy', 'cũng vậy thôi', 'cũng được', 'chỉ chính', 'chỉ có',
    'chỉ là', 'chỉ tên', 'chính bản', 'chính giữa', 'chính là',
    'chính thị', 'chính điểm', 'đúng ngày', 'đúng ra', 'đúng tuổi',
    'đúng với', 'quả là', 'quả thật', 'quả thế', 'quả vậy',
    'thực hiện', 'thực hiện đúng', 'thực ra', 'thực sự', 'thực tế',
    'thực vậy', 'thật chắc', 'thật là', 'thật lực', 'thật quả',
    'thật ra', 'thật sự', 'thật thà', 'thật tốt', 'thật vậy',
    'hẳn là', 'ắt hẳn', 'ắt là', 'ắt phải', 'ắt thật', 'chắc chắn',
    'chắc dạ', 'chắc hẳn', 'chắc lòng', 'chắc người', 'chắc vào',
    'chắc ăn', 'có ai', 'có chuyện', 'có chăng', 'có chứ', 'có cơ',
    'có dễ', 'có họ', 'có khi', 'có ngày', 'có người', 'có nhiều',
    'có nhà', 'có phải', 'có số', 'có tháng', 'có thế', 'có thể',
    'có vẻ', 'có ý', 'có ăn', 'có điều', 'có điều kiện', 'có đáng',
    'có đâu', 'có được', 'có lẽ', 'có thể', 'có khi', 'chừng như',
    'hình như', 'dường như', 'dường như là', 'như là', 'tựa như là',
    'có vẻ', 'có vẻ như', 'phải biết', 'phải chi', 'phải chăng',
    'phải cách', 'phải cái', 'phải giờ', 'phải khi', 'phải không',
    'phải lại', 'phải lời', 'phải người', 'phải như', 'phải rồi',
    'phải tay', 'cần cấp', 'cần gì', 'cần số', 'nên chi', 'nên chăng',
    'nên làm', 'nên người', 'nên tránh', 'được cái', 'được lời',
    'được nước', 'được tin', 'bị chú', 'bị vì',
    
    # Tính từ và phó từ thường gặp
    'rất', 'lắm', 'nhiều', 'ít', 'mấy', 'bao nhiêu', 'bao', 'khá', 'tương đối',
    'khá là', 'hơi', 'hơn', 'nhất', 'cùng', 'bằng', 'như', 'tựa', 'gần như',
    'hầu như', 'gần', 'xa', 'to', 'nhỏ', 'lớn', 'bé', 'cao', 'thấp',
    'dài', 'ngắn', 'rộng', 'hẹp', 'sâu', 'nông', 'nhanh', 'chậm',
    'sớm', 'muộn', 'sáng', 'tối', 'mới', 'cũ', 'trẻ', 'già',
    'rất lâu', 'nhiều ít', 'nhiều lắm', 'ít biết', 'ít có', 'ít hơn',
    'ít khi', 'ít lâu', 'ít nhiều', 'ít nhất', 'ít nữa', 'ít quá',
    'ít ra', 'ít thôi', 'ít thấy', 'bao lâu', 'bao nả', 'khá tốt',
    'khá khá', 'tương đối', 'hơi hơi', 'hơn cả', 'hơn hết', 'hơn là',
    'hơn nữa', 'hơn trước', 'nhất loạt', 'nhất luật', 'nhất là',
    'nhất mực', 'nhất nhất', 'nhất quyết', 'nhất sinh', 'nhất thiết',
    'nhất thì', 'nhất tâm', 'nhất tề', 'nhất đán', 'nhất định',
    'cùng chung', 'cùng cực', 'cùng nhau', 'cùng tuổi', 'cùng tột',
    'cùng với', 'cùng ăn', 'bằng cứ', 'bằng không', 'bằng người',
    'bằng nhau', 'bằng như', 'bằng nào', 'bằng nấy', 'bằng vào',
    'bằng được', 'bằng ấy', 'như ai', 'như chơi', 'như không',
    'như là', 'như nhau', 'như quả', 'như sau', 'như thường',
    'như thế', 'như thế nào', 'như thể', 'như trên', 'như trước',
    'như tuồng', 'như vậy', 'như ý', 'tựa như', 'gần như',
    'hầu như', 'gần bên', 'gần hết', 'gần ngày', 'gần như',
    'gần xa', 'gần đây', 'gần đến', 'xa cách', 'xa gần', 'xa nhà',
    'xa tanh', 'xa tắp', 'xa xa', 'xa xả', 'to nhỏ', 'nhỏ người',
    'lớn lên', 'lớn nhỏ', 'cao lâu', 'cao ráo', 'cao răng', 'cao sang',
    'cao số', 'cao thấp', 'cao thế', 'cao xa', 'thấp cơ', 'thấp thỏm',
    'thấp xuống', 'dài lời', 'dài ra', 'nhanh lên', 'nhanh tay',
    'sớm ngày', 'sáng ngày', 'sáng rõ', 'sáng thế', 'sáng ý',
    'mới hay', 'mới rồi', 'mới đây',
    
    # Số từ và lượng từ
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'trăm', 'nghìn', 'triệu', 'tỷ', 'vài', 'mấy', 'dăm', 'chục', 'tá',
    'đôi', 'cặp', 'đống', 'lô', 'bó', 'bọc', 'gói', 'kiện',
    'một cách', 'một cơn', 'một khi', 'một lúc', 'một số', 'một vài',
    'một ít', 'vài ba', 'vài người', 'vài nhà', 'vài nơi', 'vài tên',
    'vài điều', 'mấy cái', 'mấy thứ', 'mấy món', 'ba ba', 'ba bản',
    'ba cùng', 'ba họ', 'ba ngày', 'ba ngôi', 'ba tăng',
    
    # Từ chỉ thời gian
    'khi', 'lúc', 'hồi', 'hôm', 'ngày', 'tháng', 'năm', 'tuần', 'giờ',
    'phút', 'giây', 'sáng', 'chiều', 'tối', 'đêm', 'trưa', 'xưa',
    'nay', 'mai', 'mốt', 'kia', 'bây giờ', 'lúc này', 'hiện tại',
    'trước đây', 'sau này', 'từ nay', 'từ trước', 'cho đến',
    'đến khi', 'cho tới', 'tính từ', 'bắt đầu từ', 'khi khác',
    'khi không', 'khi nào', 'khi nên', 'khi trước', 'lúc khác',
    'lúc lâu', 'lúc nào', 'lúc này', 'lúc sáng', 'lúc trước',
    'lúc đi', 'lúc đó', 'lúc đến', 'lúc ấy', 'hồi nào', 'hôm nào',
    'ngày càng', 'ngày cấp', 'ngày giờ', 'ngày ngày', 'ngày nào',
    'ngày này', 'ngày nọ', 'ngày qua', 'ngày rày', 'ngày tháng',
    'ngày xưa', 'ngày xửa', 'ngày đến', 'ngày ấy', 'tháng ngày',
    'tháng năm', 'tháng tháng', 'năm tháng', 'tuần tự', 'giờ lâu',
    'giờ này', 'giờ đi', 'giờ đây', 'giờ đến', 'phút giây',
    'sáng chiều', 'chiều tối', 'tối đêm', 'đêm ngày', 'trưa chiều',
    'xưa nay', 'nay mai', 'mai mốt', 'kia này', 'bây chừ',
    'bây giờ', 'bây nhiêu', 'bây nay', 'bấy chầy', 'bấy chừ',
    'bấy giờ', 'bấy lâu', 'bấy lâu nay', 'bấy nay', 'bấy nhiêu',
    'hiện nay', 'hiện tại', 'cho đến', 'đến khi', 'cho tới',
    'tính từ', 'bắt đầu từ',
    
    # Từ phủ định
    'không', 'chẳng', 'chả', 'chưa', 'đừng', 'đừng có', 'không có',
    'chẳng có', 'chả có', 'chưa có', 'không bao giờ', 'chẳng bao giờ',
    'chả bao giờ', 'không ai', 'chẳng ai', 'chả ai', 'không ai',
    'không bao lâu', 'không biết', 'không bán', 'không chỉ',
    'không còn', 'không có', 'không có gì', 'không cùng',
    'không cần', 'không cứ', 'không dùng', 'không gì',
    'không hay', 'không khỏi', 'không kể', 'không ngoài',
    'không nhận', 'không những', 'không phải', 'không phải không',
    'không thể', 'không tính', 'không điều kiện', 'không được',
    'không đầy', 'không để', 'chẳng lẽ', 'chẳng những',
    'chẳng nữa', 'chẳng phải', 'chưa bao giờ', 'chưa chắc',
    'chưa có', 'chưa cần', 'chưa dùng', 'chưa dễ', 'chưa kể',
    'chưa tính', 'chưa từng', 'đừng có',
    
    # Từ khẳng định
    'có', 'được', 'rồi', 'ồ', 'ừ', 'vâng', 'dạ', 'phải', 'đúng',
    'ừm', 'uh', 'um', 'ok', 'okay', 'vâng chịu', 'vâng dạ',
    'vâng vâng', 'vâng ý', 'dạ bán', 'dạ con', 'dạ dài',
    'dạ dạ', 'dạ khách', 'rồi nữa', 'rồi ra', 'rồi sao',
    'rồi sau', 'rồi tay', 'rồi thì', 'rồi xem', 'rồi đây',
    
    # Từ cảm thán và tình thái
    'ơi', 'ạ', 'ý', 'à', 'ừ', 'hử', 'hè', 'he', 'hi', 'hô', 'ôi',
    'ối', 'úi', 'ui', 'ái', 'ay', 'dạ', 'vâng', 'thưa', 'kính thưa',
    'thật', 'thực', 'quả thật', 'quả thực', 'thật sự', 'thực sự',
    'à này', 'à ơi', 'ào', 'ào vào', 'ào ào', 'á', 'á à',
    'ái chà', 'ái dà', 'âu là', 'ô hay', 'ô hô', 'ô kê',
    'ô kìa', 'ôi chao', 'ôi thôi', 'úi chà', 'úi dào',
    'ý chừng', 'ý da', 'ý hoặc', 'ạ ơi', 'ầu ơ', 'ối dào',
    'ối giời', 'ối giời ơi', 'ồ ồ', 'ớ này', 'ờ ờ', 'ủa',
    'ứ hự', 'ứ ừ', 'ừ nhé', 'ừ thì', 'ừ ào', 'ừ ừ',
    'chao ôi', 'oai oái', 'oái', 'than ôi', 'trời đất ơi',
    'alô', 'amen', 'a lô', 'a ha', 'pho', 'phè', 'phè phè',
    'phóc', 'phót', 'phăn phắt', 'phỉ phui', 'phỏng',
    'phỏng như', 'phỏng nước', 'phỏng theo', 'phỏng tính',
    'phốc', 'phụt', 'phứt', 'ren rén', 'riu ríu', 'rích',
    'rón rén', 'sa sả', 'sì', 'sì sì', 'sất', 'sốt sột',
    'tanh', 'tanh tanh', 'tha hồ', 'tha hồ chơi', 'tha hồ ăn',
    'thoạt', 'thoạt nghe', 'thoạt nhiên', 'thoắt', 'thúng thắng',
    'thương ôi', 'thảo hèn', 'thảo nào', 'toé khói', 'toẹt',
    'trếu tráo', 'trển', 'trệt', 'trệu trạo', 'trỏng',
    'tuốt luốt', 'tuốt tuồn tuột', 'tuốt tuột', 'tà tà',
    'tênh', 'tênh tênh', 'tít mù', 'tò te', 'tông tốc',
    'tù tì', 'tăm tắp', 'tắp', 'tắp lự', 'tắp tắp',
    'tọt', 'tớ', 'tức thì', 'tức tốc', 'veo', 'veo veo',
    'vung thiên địa', 'vung tàn tán', 'vung tán tàn',
    'vèo', 'vèo vèo', 'văng tê', 'vạn nhất', 'vả chăng',
    'vả lại', 'vở', 'vụt', 'xon xón', 'xoành xoạch',
    'xoét', 'xoẳn', 'xoẹt', 'xuể', 'xềnh xệch', 'xệp',
    
    # Từ nối câu và đoạn văn
    'đầu tiên', 'thứ nhất', 'thứ hai', 'thứ ba', 'cuối cùng', 'sau cùng',
    'trước hết', 'trước tiên', 'đầu tiên', 'kế đó', 'tiếp theo', 'sau đó',
    'rồi', 'rồi thì', 'thế rồi', 'vậy rồi', 'xong', 'xong rồi',
    'mà thôi', 'thôi', 'thôi thì', 'cũng thôi', 'tiếp theo',
    'tiếp tục', 'tiếp đó', 'tiện thể', 'cuối', 'cuối cùng',
    'cuối điểm', 'cuốn', 'cuộc', 'rút cục', 'rốt cuộc',
    'rốt cục', 'chót', 'chốt', 'kết quả', 'kết cục',
    'kết luận', 'kết thúc', 'hoàn thành', 'xong xuôi',
    'ơ kìa', 'ơ hay',
    
    # Từ chỉ địa điểm (tổng quát)
    'đây', 'đó', 'kia', 'ấy', 'này', 'kìa', 'chỗ', 'nơi', 'chốn',
    'vùng', 'miền', 'vị trí', 'địa điểm', 'nơi nào', 'đâu đó',
    'nơi đây', 'nơi đó', 'nơi kia', 'bên này', 'bên kia', 'bên ấy',
    'đây này', 'đây rồi', 'đây đó', 'đó đây', 'nơi nơi',
    'chỗ nào', 'chốn nào', 'vùng lên', 'vùng nước', 'miền nào',
    'vị trí nào', 'địa điểm nào',
    
    # Từ chỉ cách thức
    'như', 'như thế', 'như vậy', 'thế nào', 'sao', 'ra sao', 'cách nào',
    'làm sao', 'bằng cách', 'theo cách', 'cách', 'kiểu', 'thể',
    'dạng', 'loại', 'loại nào', 'kiểu gì', 'thể nào', 'sao bản',
    'sao bằng', 'sao cho', 'sao vậy', 'sao đang', 'làm bằng',
    'làm cho', 'làm dần dần', 'làm gì', 'làm lòng', 'làm lại',
    'làm lấy', 'làm mất', 'làm ngay', 'làm như', 'làm nên',
    'làm ra', 'làm riêng', 'làm sao', 'làm theo', 'làm thế nào',
    'làm tin', 'làm tôi', 'làm tăng', 'làm tại', 'làm tắp lự',
    'làm vì', 'làm đúng', 'làm được', 'bằng cách nào',
    'theo cách nào', 'cách bức', 'cách không', 'cách nhau',
    'cách đều', 'kiểu gì', 'thể nào', 'dạng nào', 'loại gì',
    
    # Từ chỉ nguyên nhân
    'tại sao', 'vì sao', 'tại', 'vì', 'do', 'bởi', 'nhờ', 'nhờ có',
    'nhờ vào', 'do có', 'do vào', 'vì có', 'vì vào', 'bởi có', 'bởi vào',
    'do vì', 'do vậy', 'do đó', 'bởi ai', 'bởi chưng', 'bởi nhưng',
    'bởi sao', 'bởi thế', 'bởi thế cho nên', 'bởi tại', 'bởi vì',
    'bởi vậy', 'bởi đâu', 'nhờ chuyển', 'nhờ có', 'nhờ nhờ', 'nhờ đó',
    'tại lòng', 'tại nơi', 'tại sao', 'tại tôi', 'tại vì', 'tại đâu',
    'tại đây', 'tại đó', 'vì chưng', 'vì rằng', 'vì sao', 'vì thế',
    'vì vậy',
    
    # Từ chỉ mục đích
    'để', 'để mà', 'nhằm', 'nhằm mục đích', 'với mục đích', 'cho',
    'cho việc', 'hòng', 'mong', 'mong muốn', 'hy vọng', 'để cho',
    'để giống', 'để không', 'để lòng', 'để lại', 'để mà', 'để phần',
    'để được', 'để đến nỗi', 'nhằm khi', 'nhằm lúc', 'nhằm vào',
    'nhằm để', 'cho biết', 'cho chắc', 'cho hay', 'cho nhau',
    'cho nên', 'cho rằng', 'cho rồi', 'cho thấy', 'cho tin',
    'cho tới', 'cho tới khi', 'cho về', 'cho ăn', 'cho đang',
    'cho được', 'cho đến', 'cho đến khi', 'cho đến nỗi',
    'mong muốn', 'hy vọng',
    
    # Từ chỉ điều kiện
    'nếu', 'nếu như', 'giả sử', 'giả như', 'trong trường hợp',
    'khi mà', 'khi nào', 'bao giờ', 'lúc nào', 'hễ', 'mỗi khi',
    'mỗi lúc', 'cứ khi', 'cứ mỗi khi', 'nếu có', 'nếu cần',
    'nếu không', 'nếu mà', 'nếu như', 'nếu thế', 'nếu vậy',
    'nếu được', 'giả sử', 'giả như', 'trong trường hợp',
    'khi mà', 'khi nào', 'bao giờ', 'lúc nào', 'hễ', 'mỗi khi',
    'mỗi lúc', 'cứ khi', 'cứ mỗi khi', 'cứ như', 'cứ việc',
    'cứ điểm', 'khiến', 'ví bằng', 'ví dù', 'ví phỏng', 'ví thử',
    'nhược bằng',
    
    # Từ chỉ sự so sánh
    'hơn', 'kém', 'bằng', 'như', 'tương tự', 'giống', 'giống như',
    'tựa như', 'khác', 'khác với', 'không như', 'chẳng như',
    'so với', 'đối với', 'về phần', 'còn về', 'giống người',
    'giống nhau', 'giống như', 'khác gì', 'khác khác', 'khác nhau',
    'khác nào', 'khác thường', 'khác xa', 'so với', 'đối với',
    'về phần', 'còn về', 'hơn cả', 'hơn hết', 'hơn là',
    'hơn nữa', 'hơn trước', 'kém hơn', 'bằng cứ', 'bằng không',
    'bằng người', 'bằng nhau', 'bằng như', 'bằng nào', 'bằng nấy',
    'bằng vào', 'bằng được', 'bằng ấy', 'tương tự như',
    'tựa như', 'tựa như là',
    
    # Từ hạn định
    'các', 'những', 'mọi', 'mỗi', 'từng', 'tất cả', 'toàn bộ',
    'toàn thể', 'cả', 'cả thảy', 'tổng cộng', 'chung', 'riêng',
    'riêng biệt', 'đặc biệt', 'nói riêng', 'nói chung', 'các cậu',
    'những ai', 'những khi', 'những là', 'những lúc', 'những muốn',
    'những như', 'mọi giờ', 'mọi khi', 'mọi lúc', 'mọi người',
    'mọi nơi', 'mọi sự', 'mọi thứ', 'mọi việc', 'mỗi lúc',
    'mỗi lần', 'mỗi một', 'mỗi ngày', 'mỗi người', 'từng cái',
    'từng giờ', 'từng nhà', 'từng phần', 'từng thời gian',
    'từng đơn vị', 'từng ấy', 'tất cả bao nhiêu', 'tất thảy',
    'tất tần tật', 'tất tật', 'toàn bộ', 'toàn thể', 'cả nghe',
    'cả nghĩ', 'cả ngày', 'cả người', 'cả nhà', 'cả năm',
    'cả thảy', 'cả thể', 'cả tin', 'cả ăn', 'cả đến',
    'riêng từng', 'đặc biệt', 'nói riêng', 'nói chung',
    
    # Từ chỉ tần suất
    'thường', 'thường xuyên', 'luôn', 'luôn luôn', 'lúc nào cũng',
    'bao giờ cũng', 'mãi', 'mãi mãi', 'hoài', 'suốt', 'liên tục',
    'liên tiếp', 'khi thì', 'lúc thì', 'đôi khi', 'thỉnh thoảng',
    'thỉnh thoảng có khi', 'hiếm khi', 'ít khi', 'chưa bao giờ',
    'không bao giờ', 'chẳng bao giờ', 'thường bị', 'thường hay',
    'thường khi', 'thường số', 'thường sự', 'thường thôi',
    'thường thường', 'thường tính', 'thường tại', 'thường xuất hiện',
    'thường đến', 'luôn cả', 'luôn luôn', 'luôn tay', 'lúc nào cũng',
    'bao giờ cũng', 'mãi mãi', 'suốt đời', 'liên tục',
    'liên tiếp', 'thi thoảng', 'đôi khi', 'thỉnh thoảng',
    'thỉnh thoảng có khi', 'hiếm khi', 'ít khi', 'lâu lâu',
    'lần lần', 'lần khác', 'lần nào', 'lần này', 'lần sang',
    'lần sau', 'lần theo', 'lần trước', 'lần tìm',
    
    # Từ chỉ mức độ
    'quá', 'quá đỗi', 'quá chừng', 'quá mức', 'cực', 'cực kỳ',
    'vô cùng', 'vô vàn', 'hết sức', 'rất', 'rất là', 'lắm',
    'nhiều', 'nhiều lắm', 'ít', 'ít ỏi', 'chút', 'chút ít',
    'một chút', 'đôi chút', 'hơi', 'hơi hơi', 'khá', 'khá là',
    'tương đối', 'khá khá', 'còn', 'còn như', 'càng', 'càng ngày càng',
    'ngày càng', 'ngày một', 'từng ngày một', 'quá bán', 'quá bộ',
    'quá giờ', 'quá lời', 'quá mức', 'quá nhiều', 'quá tay',
    'quá thì', 'quá tin', 'quá trình', 'quá tuổi', 'quá đáng',
    'quá ư', 'cực lực', 'vô hình trung', 'vô kể', 'vô luận',
    'vô vàn', 'hết chuyện', 'hết cả', 'hết của', 'hết nói',
    'hết ráo', 'hết rồi', 'hết ý', 'rất lâu', 'lắm lời',
    'nhiều ít', 'nhiều lắm', 'ít ỏi', 'chút ít', 'một chút',
    'đôi chút', 'hơi hơi', 'khá là', 'khá khá', 'tương đối',
    'càng càng', 'càng hay', 'ngày càng', 'ngày một', 'từng ngày một',
    'còn như', 'còn nữa', 'còn thời gian', 'còn về',
    
    # Từ chỉ sự chắc chắn
    'chắc', 'chắc chắn', 'chắc hẳn', 'dĩ nhiên', 'tất nhiên',
    'đương nhiên', 'hiển nhiên', 'rõ ràng', 'rành rành', 'hẳn',
    'hẳn là', 'ắt', 'ắt hẳn', 'ắt là', 'có lẽ', 'có thể',
    'có khi', 'chừng như', 'hình như', 'dường như', 'dường như là',
    'như là', 'tựa như là', 'có vẻ', 'có vẻ như', 'chắc chắn',
    'chắc dạ', 'chắc hẳn', 'chắc lòng', 'chắc người', 'chắc vào',
    'chắc ăn', 'dĩ nhiên', 'tất nhiên', 'đương nhiên', 'hiển nhiên',
    'rõ ràng', 'rành rành', 'rõ là', 'rõ thật', 'hẳn là',
    'ắt hẳn', 'ắt là', 'ắt phải', 'ắt thật', 'nghiễm nhiên',
    'công nhiên', 'khẳng định', 'tuyệt nhiên',
    
    # Các từ khác thường gặp
    'gì', 'chi', 'đâu', 'nào', 'sao', 'đâu đó', 'nào đó',
    'gì đó', 'ai đó', 'điều gì', 'việc gì', 'cái gì',
    'thứ gì', 'loại gì', 'kiểu gì', 'thể nào', 'bằng cách nào',
    'bao nhiêu', 'mấy', 'mấy cái', 'mấy thứ', 'mấy món',
    'gì gì', 'gì đó', 'chi gì', 'đâu có', 'đâu cũng',
    'đâu như', 'đâu nào', 'đâu phải', 'đâu đâu', 'đâu đây',
    'đâu đó', 'nào cũng', 'nào hay', 'nào là', 'nào phải',
    'nào đâu', 'nào đó', 'điều', 'điều gì', 'điều kiện',
    'việc', 'việc gì', 'cái', 'cái gì', 'cái họ', 'cái đã',
    'cái đó', 'cái ấy', 'thứ', 'thứ bản', 'thứ đến',
    'loại', 'loại từ', 'kiểu', 'thể', 'dạng',
    
    # Các từ bổ sung từ danh sách thứ 2
    'bay biến', 'biết', 'biết bao', 'biết bao nhiêu', 'biết chắc',
    'biết chừng nào', 'biết mình', 'biết mấy', 'biết thế',
    'biết trước', 'biết việc', 'biết đâu', 'biết đâu chừng',
    'biết đâu đấy', 'biết được', 'buổi', 'buổi làm', 'buổi mới',
    'buổi ngày', 'buổi sớm', 'bài', 'bài bác', 'bài bỏ',
    'bài cái', 'bác', 'bán', 'bán cấp', 'bán dạ', 'bán thế',
    'bây bẩy', 'béng', 'bông', 'bước', 'bước khỏi', 'bước tới',
    'bước đi', 'bản', 'bản bộ', 'bản riêng', 'bản thân', 'bản ý',
    'bất chợt', 'bất cứ', 'bất giác', 'bất kì', 'bất kể', 'bất kỳ',
    'bất luận', 'bất ngờ', 'bất nhược', 'bất quá', 'bất quá chỉ',
    'bất thình lình', 'bất tử', 'bất đồ', 'bấy', 'bập bà bập bõm',
    'bập bõm', 'bắt đầu', 'bắt đầu từ', 'bển', 'bệt',
    'bỏ', 'bỏ bà', 'bỏ cha', 'bỏ cuộc', 'bỏ không', 'bỏ lại',
    'bỏ mình', 'bỏ mất', 'bỏ mẹ', 'bỏ nhỏ', 'bỏ quá', 'bỏ ra',
    'bỏ riêng', 'bỏ việc', 'bỏ xa', 'bỗng', 'bỗng chốc',
    'bỗng dưng', 'bỗng không', 'bỗng nhiên', 'bỗng nhưng',
    'bỗng thấy', 'bỗng đâu', 'bộ', 'bộ thuộc', 'bộ điều',
    'bội phần', 'bớ', 'bức', 'cao', 'cha', 'cha chả',
    'chia sẻ', 'chiếc', 'chu cha', 'chui cha', 'chung cho',
    'chung chung', 'chung cuộc', 'chung cục', 'chung nhau',
    'chung qui', 'chung quy', 'chung quy lại', 'chung ái',
    'chuyển', 'chuyển tự', 'chuyển đạt', 'chuyện', 'chuẩn bị',
    'chành chạnh', 'chí chết', 'chùn chùn', 'chùn chũn',
    'chăn chắn', 'chăng', 'chăng chắc', 'chăng nữa', 'chơi',
    'chơi họ', 'chầm chập', 'chậc', 'chẳng lẽ', 'chẳng những',
    'chẳng nữa', 'chẳng phải', 'chết nỗi', 'chết thật', 'chết tiệt',
    'chỉn', 'chịu', 'chịu chưa', 'chịu lời', 'chịu tốt', 'chịu ăn',
    'chọn', 'chọn bên', 'chọn ra', 'chốc chốc', 'chớ', 'chớ chi',
    'chớ gì', 'chớ không', 'chớ kể', 'chớ như', 'chợt', 'chợt nghe',
    'chợt nhìn', 'chủn', 'chứ', 'chứ ai', 'chứ còn', 'chứ gì',
    'chứ không', 'chứ không phải', 'chứ lại', 'chứ lị', 'chứ như',
    'chứ sao', 'coi bộ', 'coi mòi', 'con con', 'con tính',
    'cách bức', 'cách không', 'cách nhau', 'cách đều', 'câu hỏi',
    'cây', 'cây nước', 'cóc khô', 'căn', 'căn cái', 'căn cắt',
    'căn tính', 'cơ', 'cơ chỉ', 'cơ chừng', 'cơ cùng', 'cơ dẫn',
    'cơ hồ', 'cơ hội', 'cơ mà', 'cơn', 'cảm thấy', 'cảm ơn',
    'cấp', 'cấp số', 'cấp trực tiếp', 'cật lực', 'cật sức',
    'cổ lai', 'cụ thể', 'cụ thể là', 'cụ thể như', 'của ngọt',
    'của tin', 'duy', 'duy chỉ', 'duy có', 'dành', 'dành dành',
    'dào', 'dì', 'dần dà', 'dần dần', 'dầu sao', 'dẫn',
    'dễ', 'dễ dùng', 'dễ gì', 'dễ khiến', 'dễ nghe', 'dễ ngươi',
    'dễ như chơi', 'dễ sợ', 'dễ sử dụng', 'dễ thường', 'dễ thấy',
    'dễ ăn', 'dễ đâu', 'dở chừng', 'dữ', 'dữ cách', 'em em',
    'giá trị', 'giá trị thực tế', 'giảm', 'giảm chính', 'giảm thấp',
    'giảm thế', 'giữ', 'giữ lấy', 'giữ ý', 'gây', 'gây cho',
    'gây giống', 'gây ra', 'gây thêm', 'gồm', 'hay biết', 'hay hay',
    'hay không', 'hay là', 'hay làm', 'hay nhỉ', 'hay nói', 'hay sao',
    'hay tin', 'hay đâu', 'hiểu', 'hầu hết', 'hết', 'hỏi',
    'hỏi lại', 'hỏi xem', 'hỏi xin', 'hỗ trợ', 'khoảng',
    'khoảng cách', 'khoảng không', 'khách', 'khó', 'khó biết',
    'khó chơi', 'khó khăn', 'khó làm', 'khó mở', 'khó nghe',
    'khó nghĩ', 'khó nói', 'khó thấy', 'khó tránh', 'khỏi',
    'khỏi nói', 'kể', 'kể cả', 'kể như', 'kể tới', 'kể từ',
    'liên quan', 'lâu', 'lâu các', 'lâu lâu', 'lâu nay',
    'lâu ngày', 'lòng', 'lòng không', 'lý do', 'lượng',
    'lượng cả', 'lượng số', 'lượng từ', 'lại', 'lại bộ',
    'lại cái', 'lại còn', 'lại giống', 'lại làm', 'lại người',
    'lại nói', 'lại nữa', 'lại quả', 'lại thôi', 'lại ăn',
    'lại đây', 'lấy', 'lấy có', 'lấy cả', 'lấy giống', 'lấy làm',
    'lấy lý do', 'lấy lại', 'lấy ra', 'lấy ráo', 'lấy sau',
    'lấy số', 'lấy thêm', 'lấy thế', 'lấy vào', 'lấy xuống',
    'lấy được', 'lấy để', 'lời', 'lời chú', 'lời nói',
    'mang', 'mang lại', 'mang mang', 'mang nặng', 'mang về',
    'muốn', 'mạnh', 'mất', 'mất còn', 'mối', 'mở', 'mở mang',
    'mở nước', 'mở ra', 'mức', 'ngay', 'ngay bây giờ', 'ngay cả',
    'ngay khi', 'ngay khi đến', 'ngay lúc', 'ngay lúc này',
    'ngay lập tức', 'ngay thật', 'ngay tức khắc', 'ngay tức thì',
    'ngay từ', 'nghe', 'nghe chừng', 'nghe hiểu', 'nghe không',
    'nghe lại', 'nghe nhìn', 'nghe như', 'nghe nói', 'nghe ra',
    'nghe rõ', 'nghe thấy', 'nghe tin', 'nghe trực tiếp', 'nghe đâu',
    'nghe đâu như', 'nghe được', 'nghen', 'nghĩ', 'nghĩ lại',
    'nghĩ ra', 'nghĩ tới', 'nghĩ xa', 'nghĩ đến', 'nghỉm',
    'ngoải', 'nguồn', 'ngôi', 'ngôi nhà', 'ngôi thứ', 'ngõ hầu',
    'ngăn ngắt', 'người hỏi', 'người khác', 'người khách',
    'người mình', 'người nghe', 'người người', 'người nhận',
    'ngọn', 'ngọn nguồn', 'ngọt', 'ngồi', 'ngồi bệt', 'ngồi không',
    'ngồi sau', 'ngồi trệt', 'ngộ nhỡ', 'nhung nhăng', 'nhà',
    'nhà chung', 'nhà khó', 'nhà làm', 'nhà ngoài', 'nhà ngươi',
    'nhà tôi', 'nhà việc', 'nhân dịp', 'nhân tiện', 'nhé',
    'nhìn', 'nhìn chung', 'nhìn lại', 'nhìn nhận', 'nhìn theo',
    'nhìn thấy', 'nhìn xuống', 'nhóm', 'nhón nhén', 'nhược bằng',
    'nhận', 'nhận biết', 'nhận họ', 'nhận làm', 'nhận nhau',
    'nhận ra', 'nhận thấy', 'nhận việc', 'nhận được', 'nhỉ',
    'nhỏ', 'nhớ', 'nhớ bập bõm', 'nhớ lại', 'nhớ lấy', 'nhớ ra',
    'nhỡ ra', 'nóc', 'nói', 'nói bông', 'nói chung', 'nói khó',
    'nói là', 'nói lên', 'nói lại', 'nói nhỏ', 'nói phải', 'nói qua',
    'nói ra', 'nói riêng', 'nói rõ', 'nói thêm', 'nói thật',
    'nói toẹt', 'nói trước', 'nói tốt', 'nói với', 'nói xa',
    'nói ý', 'nói đến', 'nói đủ', 'nấy', 'nặng', 'nặng căn',
    'nặng mình', 'nặng về', 'nền', 'nọ', 'nớ', 'nức nở',
    'nữa', 'nữa khi', 'nữa là', 'nữa rồi', 'phía', 'phía bên',
    'phía bạn', 'phía dưới', 'phía sau', 'phía trong', 'phía trên',
    'phía trước', 'phù hợp', 'phương chi',     'phần', 'phần lớn',
    'phần nhiều', 'phần nào', 'phần sau', 'phần việc', 'phắt',
    'phỉ phui', 'phỏng', 'phỏng như', 'phỏng nước', 'phỏng theo',
    'phỏng tính', 'phốc', 'phụt', 'phứt', 'qua', 'quan trọng',
    'quan trọng vấn đề', 'quan tâm', 'quay', 'quay bước', 'quay lại',
    'quay số', 'quay đi', 'quận', 'riệt', 'rày', 'ráo',
    'ráo cả', 'ráo nước', 'ráo trọi', 'rén', 'rén bước',
    'rút cục', 'răng', 'răng răng', 'rồi', 'rứa', 'sang',
    'sốt sột', 'sớm', 'sở dĩ', 'sử dụng', 'sự', 'sự thế',
    'sự việc', 'tha hồ', 'thanh', 'thanh ba', 'thanh chuyển',
    'thanh không', 'thanh thanh', 'thanh tính', 'thanh điều kiện',
    'thanh điểm', 'thiếu', 'thiếu gì', 'thiếu điểm', 'thuần',
    'thuần ái', 'thuộc', 'thuộc bài', 'thuộc cách', 'thuộc lại',
    'thuộc từ', 'thà', 'thà là', 'thà rằng', 'thái quá',
    'thêm', 'thêm chuyện', 'thêm giờ', 'thêm vào', 'thình lình',
    'thích', 'thích cứ', 'thích thuộc', 'thích tự', 'thích ý',
    'thôi', 'thôi việc', 'thế', 'thế chuẩn bị', 'thế là',
    'thế lại', 'thế mà', 'thế nào', 'thế nên', 'thế ra',
    'thế sự', 'thế thì', 'thế thôi', 'thế thường', 'thế thế',
    'thế à', 'thế đó', 'thếch', 'thỉnh thoảng', 'thỏm',
    'thốc', 'thốc tháo', 'thốt', 'thốt nhiên', 'thốt nói',
    'thốt thôi', 'thộc', 'thời gian', 'thời gian sử dụng',
    'thời gian tính', 'thời điểm', 'thục mạng', 'thửa',
    'thực hiện', 'thực hiện đúng', 'thực ra', 'thực sự',
    'thực tế', 'thực vậy', 'tin', 'tin thêm', 'tin vào',
    'tránh', 'tránh khỏi', 'tránh ra', 'tránh tình trạng',
    'tránh xa', 'trả', 'trả của', 'trả lại', 'trả ngay',
    'trả trước', 'trở thành', 'trừ phi', 'trực tiếp',
    'trực tiếp làm', 'tuy', 'tuy có', 'tuy là', 'tuy nhiên',
    'tuy rằng', 'tuy thế', 'tuy vậy', 'tuy đã', 'tuổi',
    'tuổi cả', 'tuổi tôi', 'tên', 'tên chính', 'tên cái',
    'tên họ', 'tên tự', 'tìm', 'tìm bạn', 'tìm cách',
    'tìm hiểu', 'tìm ra', 'tìm việc', 'tình trạng', 'tính',
    'tính cách', 'tính căn', 'tính người', 'tính phỏng',
    'tính từ', 'tôi', 'tôi con', 'tạo', 'tạo cơ hội',
    'tạo nên', 'tạo ra', 'tạo ý', 'tạo điều kiện', 'tấm',
    'tấm bản', 'tấm các', 'tấn', 'tấn tới', 'tập trung',
    'tỏ ra', 'tỏ vẻ', 'tốc tả', 'tối ư', 'tốt', 'tốt bạn',
    'tốt bộ', 'tốt hơn', 'tốt mối', 'tốt ngày', 'tột',
    'tột cùng', 'tự', 'tự cao', 'tự khi', 'tự lượng',
    'tự tính', 'tự tạo', 'tự vì', 'tự ý', 'tự ăn',
    'tựu trung', 'vài', 'vào', 'vèo', 'vì', 'ví bằng',
    'ví dù', 'ví phỏng', 'ví thử', 'vô hình trung',
    'vô kể', 'vô luận', 'vô vàn', 'vùng', 'vùng lên',
    'vùng nước', 'vượt', 'vượt khỏi', 'vượt quá', 'vị trí',
    'vị tất', 'vốn dĩ', 'vớ', 'vừa', 'vừa khi', 'vừa lúc',
    'vừa mới', 'vừa qua', 'vừa rồi', 'vừa vừa', 'xem',
    'xem lại', 'xem ra', 'xem số', 'xin', 'xin gặp',
    'xin vâng', 'xiết bao', 'xuất hiện', 'xuất kì bất ý',
    'xuất kỳ bất ý', 'xử lý', 'yêu cầu', 'àng', 'áng như',
    'ăn', 'ăn chung', 'ăn chắc', 'ăn chịu', 'ăn cuộc',
    'ăn hết', 'ăn hỏi', 'ăn làm', 'ăn người', 'ăn ngồi',
    'ăn quá', 'ăn riêng', 'ăn sáng', 'ăn tay', 'ăn trên',
    'ăn về', 'đáng', 'đáng kể', 'đáng lí', 'đáng lý',
    'đáng lẽ', 'đáng số', 'đánh giá', 'đánh đùng', 'đáo để',
    'đành đạch', 'đơn vị', 'đưa', 'đưa cho', 'đưa chuyện',
    'đưa em', 'đưa ra', 'đưa tay', 'đưa tin', 'đưa tới',
    'đưa vào', 'đưa về', 'đưa xuống', 'đưa đến', 'được',
    'đại loại', 'đại nhân', 'đại phàm', 'đại để', 'đạt',
    'đảm bảo', 'đầu tiên', 'đầy', 'đầy năm', 'đầy phè',
    'đầy tuổi', 'đặc biệt', 'đặt', 'đặt làm', 'đặt mình',
    'đặt mức', 'đặt ra', 'đặt trước', 'đặt để', 'đồng thời',
    'đủ', 'đủ dùng', 'đủ nơi', 'đủ số', 'đủ điều', 'đủ điểm',
    'ơ', 'ở', 'ử'
        ]

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFC', text)
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_single_model_recommendations(self, title, model_name, top_k=10):
        try:
            idx = self.indices.get(title)
            if idx is None:
                return []
            
            sim_matrix = self.similarity_matrices.get(model_name)
            if sim_matrix is None:
                return []
            
            scores = list(enumerate(sim_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
            
            recommendations = []
            for rank, (recipe_idx, similarity_score) in enumerate(scores):
                recipe = self.df.iloc[recipe_idx].to_dict()
                recipe['similarity_score'] = similarity_score
                recipe['rank'] = rank + 1
                recipe['model'] = model_name
                recommendations.append(recipe)
            
            return recommendations
        except Exception as e:
            print(f"Lỗi khi lấy gợi ý từ mô hình {model_name}: {str(e)}")
            return []

    def _weighted_borda_voting(self, model_results):  # tính điểm để reccomend món ăn, có ảnh hưởng từ trọng số của từng mô hình 
        weighted_borda_scores = defaultdict(float)
        
        for model_name, recommendations in model_results.items():
            weight = self.model_weights.get(model_name, 1.0)
            max_rank = len(recommendations)
            
            for rec in recommendations:
                borda_score = max_rank - rec['rank'] + 1
                similarity_bonus = rec['similarity_score'] * 10
                final_score = weight * (borda_score + similarity_bonus)
                weighted_borda_scores[rec['title']] += final_score  
        
        return dict(weighted_borda_scores) # tính điểm Borda của mỗi món là tổng điểm của mỗi món ăn rồi cộng vào 

    def get_similar_recipes_with_majority_voting(self, title, top_k=5):
        all_recommendations = {}
        model_results = {}
        
        for model_name, sim_matrix in self.similarity_matrices.items():
            recommendations = self.get_single_model_recommendations(title, model_name, top_k=15)
            model_results[model_name] = recommendations
        
        final_scores = self._weighted_borda_voting(model_results)
        sorted_recipes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        for recipe_title, final_score in sorted_recipes[:top_k]:
            recipe = None
            for model_name, recs in model_results.items():
                for rec in recs:
                    if rec['title'] == recipe_title:
                        recipe = rec.copy()
                        break
                if recipe:
                    break
            
            if recipe:
                recipe['consensus_score'] = final_score
                final_recommendations.append(recipe)
        
        return final_recommendations

    def get_similar_recipes(self, title, model_name='doc2vec', top_k=5):
        return self.get_similar_recipes_with_majority_voting(title, top_k)

    def search_recipes_by_name(self, query, top_k=5):
        query = self.clean_text(query) 
        scores = []
        
        for i, title in enumerate(self.df['title']):
            clean_title = self.clean_text(title)
            
            if query == clean_title:
                score = 100
            elif query in clean_title:
                score = 80
            elif all(word in clean_title for word in query.split()):
                score = 60
            elif any(word in clean_title for word in query.split()):
                words_in_common = sum(1 for word in query.split() if word in clean_title)
                total_words = len(query.split())
                score = 40 * (words_in_common / total_words)
            else:
                score = 0
            
            scores.append((i, score))
        
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        result_indices = [i for i, score in scores[:top_k] if score > 0]
        
        if result_indices:
            return self.df.iloc[result_indices].to_dict(orient='records')
        else:
            return []
     #ưu tiên món ăn có nguyên liệu và món ăn chính xác 
    def filter_recipes_by_time_and_ingredients(self, time_range=None, ingredients=None, top_k=5):
        filtered_df = self.df.copy()
        
        if time_range:  #nếu chưa có thời gian chính xác thì chấp nhận hiển thị món ăn trễ 25% thời gian gốc 
            if isinstance(time_range, int):
                specific_time = time_range
                filtered_df['time_diff'] = abs(filtered_df['readyInMinutes'] - specific_time)
                max_time_diff = max(5, int(specific_time * 0.25))
                filtered_df = filtered_df[filtered_df['time_diff'] <= max_time_diff]
                filtered_df = filtered_df.sort_values(by='time_diff')
            elif time_range == '0-15':
                filtered_df = filtered_df[filtered_df['readyInMinutes'] <= 15]
            elif time_range == '16-30':
                filtered_df = filtered_df[(filtered_df['readyInMinutes'] > 15) & (filtered_df['readyInMinutes'] <= 30)]
            elif time_range == '>30':
                filtered_df = filtered_df[filtered_df['readyInMinutes'] > 30]
        
        if ingredients:
            if isinstance(ingredients, str):
                ingredients = [ing.strip().lower() for ing in ingredients.split(',')]
            elif not isinstance(ingredients, list):
                ingredients = [str(ingredients).lower()]
            
            filtered_df['match_score'] = 0
            filtered_df['found_ingredients'] = [[] for _ in range(len(filtered_df))]
            
            for ingredient in ingredients:
                ingredient_lowercase = ingredient.lower()
                pattern = r'\b' + re.escape(ingredient_lowercase) + r'\b'
                
                try:
                    ingredients_mask = filtered_df['ingredients'].str.lower().str.contains(pattern, regex=True, na=False)
                    title_mask = filtered_df['title'].str.lower().str.contains(pattern, regex=True, na=False)
                    combined_mask = ingredients_mask | title_mask
                    
                    for idx in filtered_df[combined_mask].index:
                        filtered_df.at[idx, 'found_ingredients'].append(ingredient)
                        
                    filtered_df.loc[ingredients_mask, 'match_score'] += 1
                    filtered_df.loc[title_mask, 'match_score'] += 2
                    
                except Exception as e:
                    print(f"Error filtering for ingredient '{ingredient}': {e}")
            
            filtered_df = filtered_df[filtered_df['found_ingredients'].apply(lambda x: all(ingredient in x for ingredient in ingredients))]
            
            if filtered_df.empty and ingredients:
                filtered_df = self.df.copy()
                
                if time_range:
                    if isinstance(time_range, int):
                        specific_time = time_range
                        filtered_df['time_diff'] = abs(filtered_df['readyInMinutes'] - specific_time)
                        max_time_diff = max(5, int(specific_time * 0.25))
                        filtered_df = filtered_df[filtered_df['time_diff'] <= max_time_diff]
                        filtered_df = filtered_df.sort_values(by='time_diff')
                    elif time_range == '0-15':
                        filtered_df = filtered_df[filtered_df['readyInMinutes'] <= 15]
                    elif time_range == '16-30':
                        filtered_df = filtered_df[(filtered_df['readyInMinutes'] > 15) & (filtered_df['readyInMinutes'] <= 30)]
                    elif time_range == '>30':
                        filtered_df = filtered_df[filtered_df['readyInMinutes'] > 30]
                
                filtered_df['match_score'] = 0
                filtered_df['found_ingredients'] = [[] for _ in range(len(filtered_df))]
                
                for ingredient in ingredients:
                    ingredient_lowercase = ingredient.lower()
                    pattern = r'\b' + re.escape(ingredient_lowercase) + r'\b'
                    
                    try:
                        ingredients_mask = filtered_df['ingredients'].str.lower().str.contains(pattern, regex=True, na=False)
                        title_mask = filtered_df['title'].str.lower().str.contains(pattern, regex=True, na=False)
                        combined_mask = ingredients_mask | title_mask
                        
                        for idx in filtered_df[combined_mask].index:
                            filtered_df.at[idx, 'found_ingredients'].append(ingredient)
                        
                        filtered_df.loc[ingredients_mask, 'match_score'] += 1
                        filtered_df.loc[title_mask, 'match_score'] += 2
                    except Exception as e:
                        print(f"Error filtering for ingredient '{ingredient}': {e}")
                
                filtered_df = filtered_df[filtered_df['found_ingredients'].apply(len) > 0]
            
            if not filtered_df.empty:
                filtered_df['ingredients_count'] = filtered_df['found_ingredients'].apply(len)
                filtered_df = filtered_df.sort_values(by=['ingredients_count', 'match_score'], ascending=False)
        
        if filtered_df.empty:
            return []
        
        for col in ['match_score', 'found_ingredients', 'ingredients_count', 'time_diff']:
            if col in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=[col])
        
        result = filtered_df.head(top_k).copy()
        
        if 'readyInMinutes' in result.columns:
            result['readyInMinutes'] = result['readyInMinutes'].fillna(0).astype(int)
        
        try:
            result_dict = result.to_dict(orient='records')
            for recipe in result_dict:
                if 'title' not in recipe or not recipe['title']:
                    recipe['title'] = 'Món ăn không tên'
                if 'ingredients' not in recipe or not recipe['ingredients']:
                    recipe['ingredients'] = 'Không có thông tin nguyên liệu'
                if 'instructions' not in recipe or not recipe['instructions']:
                    recipe['instructions'] = 'Không có hướng dẫn nấu ăn'
                if 'readyInMinutes' not in recipe or not recipe['readyInMinutes']:
                    recipe['readyInMinutes'] = 30
            return result_dict
        except Exception as e:
            print(f"Error converting to dictionary: {e}")
            return []
    
    def find_recipes_with_preference(self, preference, top_k=5):
        preference = self.clean_text(preference)
        vectorizer = TfidfVectorizer(stop_words=self.vietnamese_stopwords)
        tfidf_matrix = vectorizer.fit_transform(self.df['soup'].apply(self.clean_text))
        preference_vector = vectorizer.transform([preference])
        cosine_similarities = cosine_similarity(preference_vector, tfidf_matrix)[0]
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        return self.df.iloc[top_indices].to_dict(orient='records')
    
    def get_popular_recipes(self, top_k=5):
        result = self.df.sort_values(by='readyInMinutes').head(top_k)
        
        if 'readyInMinutes' in result.columns:
            result['readyInMinutes'] = result['readyInMinutes'].fillna(0).astype(int)
        
        try:
            result_dict = result.to_dict(orient='records')
            for recipe in result_dict:
                if 'title' not in recipe or not recipe['title']:
                    recipe['title'] = 'Món ăn không tên'
                if 'ingredients' not in recipe or not recipe['ingredients']:
                    recipe['ingredients'] = 'Không có thông tin nguyên liệu'
                if 'instructions' not in recipe or not recipe['instructions']:
                    recipe['instructions'] = 'Không có hướng dẫn nấu ăn'
                if 'readyInMinutes' not in recipe or not recipe['readyInMinutes']:
                    recipe['readyInMinutes'] = 30
            return result_dict
        except Exception as e:
            print(f"Error in get_popular_recipes: {e}")
            return []

    def recommend_by_cooking_time(self, cooking_time, top_k=5):
        try:
            if isinstance(cooking_time, str):
                cooking_time = int(re.search(r'\d+', cooking_time).group())
            
            df_with_diff = self.df.copy()
            df_with_diff['time_diff'] = abs(df_with_diff['readyInMinutes'] - cooking_time)
            result = df_with_diff.sort_values(by='time_diff').head(top_k)
            
            if 'readyInMinutes' in result.columns:
                result['readyInMinutes'] = result['readyInMinutes'].fillna(0).astype(int)
            
            return result.to_dict(orient='records')
        except Exception as e:
            print(f"Lỗi khi gợi ý món ăn theo thời gian: {str(e)}")
            return []

    def find_similar_recipe_by_text(self, text, top_k=5):
        try:
            text = self.clean_text(text)
            vectorizer = TfidfVectorizer(stop_words=self.vietnamese_stopwords)
            tfidf_matrix = vectorizer.fit_transform(self.df['soup'].apply(self.clean_text))
            text_vector = vectorizer.transform([text])
            cosine_similarities = cosine_similarity(text_vector, tfidf_matrix)[0]
            top_indices = cosine_similarities.argsort()[-top_k:][::-1]
            result = self.df.iloc[top_indices].copy()
            
            if 'readyInMinutes' in result.columns:
                result['readyInMinutes'] = result['readyInMinutes'].fillna(0).astype(int)
            
            return result.to_dict(orient='records')
        except Exception as e:
            print(f"Lỗi khi tìm món ăn tương tự theo text: {str(e)}")
            return []